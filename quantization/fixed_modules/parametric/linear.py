from torch import tensor, sqrt
from torch.autograd import Variable
from torch.nn import Linear, BatchNorm1d
from torch.nn.functional import relu, linear
from torch_operation_counter import OperationsCounterMode

from quantization.fixed_modules.base_module import QParametricModule
from quantization.base_parameter import QuantizationParameters, MinMaxRangesQuantizationParameters
from quantization.functional import torch_quantize, define_quantization_ranges



def plot_smooth_hist(tensor):
    import seaborn as sns; import matplotlib.pyplot as plt
    data = (tensor.cpu() if tensor.is_cuda else tensor).numpy().flatten()
    sns.set(style='whitegrid'); sns.histplot(data, kde=True, alpha=0.8); plt.show()


class QLinear(QParametricModule):
    def __init__(self,
                 in_features,
                 out_features,
                 qi,
                 qo,
                 num_bits,
                 bias=True,
                 is_signed=False,
                 quantize_per="tensor",
                 ):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits[:2], is_signed=is_signed, quantize_per=quantize_per)
        self.in_features, self.out_features = in_features, out_features
        self.linear_module = Linear(in_features, out_features, bias=bias)

        self.quantized_weight = None
        self.quantized_bias = None

        self.register_buffer("M", tensor([], requires_grad=False))
        self.qw = QuantizationParameters(num_bits=num_bits[2], is_signed=True, quantize_per=quantize_per)
        self.num_bits = num_bits
        self.is_signed = is_signed
        self.quantize_per = quantize_per

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.linear_module.reset_parameters()

        self.quantized_weight = None
        self.quantized_bias = None

        self.qw.reset_parameters()
        self.M.data = tensor([], requires_grad=False)

    def freeze(self, qi=None, qo=None):
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = self.qw.scale * self.qi.scale / self.qo.scale

        self.quantized_weight = self.qw.quantize(self.linear_module.weight)
        self.quantized_weight = (self.quantized_weight - self.qw.zero_point).round()

        qi_scale = self.qi.scale if self.qi is not None else 1.0
        qw_scale = self.qw.scale if self.qw is not None else 1.0
        fused_scale = qi_scale * qw_scale
        qmin, qmax = define_quantization_ranges(32, signed=True)
        if self.linear_module.bias is not None:
            self.quantized_bias = torch_quantize(self.linear_module.bias, scale=fused_scale, zero_point=0, qmin=qmin, qmax=qmax)

    def forward(self, x):
        x = self.linear_module(x)
        return x

    def simulated_quantize_forward(self, x):
        if self.qi is not None:
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)

        weight, bias = self.linear_module.weight, self.linear_module.bias
        self.qw.calibrate(weight)
        weight = self.qw.fake_quantize(weight)
        self.linear_module.weight.data = weight.data
        if self.linear_module.bias is not None:
            self.linear_module.bias.data = bias.data

        x = self.linear_module(x)

        if self.qo is not None:
            self.qo.calibrate(x)
            x = self.qo.fake_quantize(x)

        return x

    def quantize_inference(self, x):
        qmin, qmax = define_quantization_ranges(self.num_bits[1], signed=self.is_signed)
        x = x - self.qi.zero_point
        x = linear(x, self.quantized_weight, self.quantized_bias)
        x = self.M * x
        x = x.round()
        x = x + self.qo.zero_point
        x = x.clamp(qmin, qmax)
        return x

    def estimated_bit_operation_precision(self, x):
        bit_widths = []
        if self.qi is not None:
            bit_widths += [self.qi.num_bits]
        if self.qo is not None:
            bit_widths += [self.qo.num_bits]
        if self.qw is not None:
            bit_widths += [self.qw.num_bits]
        expected_bit_width = sum(bit_widths) / len(bit_widths) if len(bit_widths) > 0 else 0
        with OperationsCounterMode(self) as ops_counter:
            self.linear_module(x)
        return ops_counter.total_main_operation * expected_bit_width


class QLinearBatchNormReLU(QParametricModule):
    def __init__(self,
                 in_features,
                 out_features,
                 qi,
                 qo,
                 num_bits,
                 bias=True,
                 bn_affine=True,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 is_signed=False,
                 quantize_per="tensor",
                 ):
        super(QLinearBatchNormReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits[:2], is_signed=is_signed, quantize_per=quantize_per)
        self.in_features, self.out_features = in_features, out_features
        self.linear_module = Linear(in_features, out_features, bias=bias)
        self.batch_norm_module = BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum, affine=bn_affine)

        self.quantized_weight = None
        self.quantized_bias = None

        self.qw = QuantizationParameters(num_bits=num_bits[2], is_signed=True, quantize_per=quantize_per)
        self.register_buffer("M", tensor([], requires_grad=False))
        self.num_bits = num_bits
        self.is_signed = is_signed
        self.quantize_per = quantize_per

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.linear_module.reset_parameters()
        self.batch_norm_module.reset_parameters()

        self.quantized_weight = None
        self.quantized_bias = None

        self.qw.reset_parameters()
        self.M.data = tensor([], requires_grad=False)

    def fold_bn(self, mean, std):
        if self.batch_norm_module.affine:
            gamma_ = self.batch_norm_module.weight / std
            weight = self.linear_module.weight * gamma_.view(self.out_features, 1)
            if self.linear_module.bias is not None:
                bias = gamma_ * self.linear_module.bias - gamma_ * mean + self.batch_norm_module.bias
            else:
                bias = self.batch_norm_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.linear_module.weight * gamma_.view(self.out_features, 1)
            if self.linear_module.bias is not None:
                bias = gamma_ * self.linear_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
        return weight, bias

    def forward(self, x):
        x = self.linear_module(x)
        x = self.batch_norm_module(x)
        x = relu(x)
        return x

    def simulated_quantize_forward(self, x):
        if self.qi is not None:
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)

        if self.training:
            y = self.linear_module(x)
            y = y.permute(1, 0)
            y = y.contiguous().view(self.out_features, -1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.batch_norm_module.running_mean = (1 - self.batch_norm_module.momentum) * self.batch_norm_module.running_mean + self.batch_norm_module.momentum * mean
            self.batch_norm_module.running_var = (1 - self.batch_norm_module.momentum) * self.batch_norm_module.running_var + self.batch_norm_module.momentum * var
        else:
            mean = Variable(self.batch_norm_module.running_mean)
            var = Variable(self.batch_norm_module.running_var)

        std = sqrt(var + self.batch_norm_module.eps)
        weight, bias = self.fold_bn(mean, std)

        self.qw.calibrate(weight)
        weight = self.qw.fake_quantize(weight)

        x = linear(x,
                   weight,
                   bias,
                   )
        x = relu(x)

        if self.qo is not None:
            self.qo.calibrate(x)
            x = self.qo.fake_quantize(x)
        return x

    def freeze(self, qi=None, qo=None):
        if self.quantize_per != 'tensor':
            raise NotImplementedError("Only tensor-wise quantization is supported for now.")
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = self.qw.scale * self.qi.scale / self.qo.scale

        std = sqrt(self.batch_norm_module.running_var + self.batch_norm_module.eps)
        weight, bias = self.fold_bn(self.batch_norm_module.running_mean, std)
        self.quantized_weight = self.qw.quantize(weight.data) - self.qw.zero_point

        qi_scale = self.qi.scale if self.qi is not None else 1.0
        qw_scale = self.qw.scale if self.qw is not None else 1.0
        fused_scale = qi_scale * qw_scale
        qmin, qmax = define_quantization_ranges(32, signed=True)
        self.quantized_bias = torch_quantize(bias, scale=fused_scale, zero_point=0, qmin=qmin, qmax=qmax)

    def quantize_inference(self, x):
        qmin, qmax = define_quantization_ranges(self.num_bits[1], signed=self.is_signed)
        x = x - self.qi.zero_point
        x = linear(x,
                   self.quantized_weight,
                   self.quantized_bias,
                   )
        x = self.M * x
        x = x.round()
        x = x + self.qo.zero_point
        x = x.clamp(qmin, qmax)
        return x


class QMinMaxRangesLinear(QLinear):
    # Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, 2020
    def __init__(self,
                 in_features,
                 out_features,
                 qi,
                 qo,
                 num_bits,
                 bias=True,
                 is_signed=False,
                 quantize_per="tensor",
                 quant_use_momentum=True,
                 quant_momentum=0.0,
                 quant_percentile=1.0,
                 quant_sample_ratio=1.0,
                 ):
        super(QMinMaxRangesLinear, self).__init__(in_features=in_features, out_features=out_features, qi=qi, qo=qo, num_bits=num_bits, bias=bias, is_signed=is_signed, quantize_per=quantize_per)
        self.qi = MinMaxRangesQuantizationParameters(num_bits=num_bits[0], is_signed=is_signed, use_momentum=quant_use_momentum, momentum=quant_momentum, percentile=quant_percentile, sample_ratio=quant_sample_ratio) if qi else None
        self.qo = MinMaxRangesQuantizationParameters(num_bits=num_bits[1], is_signed=is_signed, use_momentum=quant_use_momentum, momentum=quant_momentum, percentile=quant_percentile, sample_ratio=quant_sample_ratio) if qo else None
        self.qw = MinMaxRangesQuantizationParameters(num_bits=num_bits[2], is_signed=True, use_momentum=quant_use_momentum, momentum=quant_momentum, percentile=quant_percentile, sample_ratio=quant_sample_ratio)


class QMinMaxRangesLinearBatchNormReLU(QLinearBatchNormReLU):
    # Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, 2020
    def __init__(self,
                 in_features,
                 out_features,
                 qi,
                 qo,
                 num_bits,
                 bias=True,
                 bn_affine=True,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 is_signed=False,
                 quantize_per="tensor",
                 quant_use_momentum=True,
                 quant_momentum=0.0,
                 quant_percentile=1.0,
                 quant_sample_ratio=1.0,
                 ):
        super(QMinMaxRangesLinearBatchNormReLU, self).__init__(in_features=in_features, out_features=out_features, qi=qi, qo=qo, num_bits=num_bits, bias=bias, bn_affine=bn_affine, bn_eps=bn_eps, bn_momentum=bn_momentum, is_signed=is_signed, quantize_per=quantize_per)
        self.qi = MinMaxRangesQuantizationParameters(num_bits=num_bits[0], is_signed=is_signed, use_momentum=quant_use_momentum, momentum=quant_momentum, percentile=quant_percentile, sample_ratio=quant_sample_ratio) if qi else None
        self.qo = MinMaxRangesQuantizationParameters(num_bits=num_bits[1], is_signed=False, use_momentum=quant_use_momentum, momentum=quant_momentum, percentile=quant_percentile, sample_ratio=quant_sample_ratio) if qo else None
        self.qw = MinMaxRangesQuantizationParameters(num_bits=num_bits[2], is_signed=True, use_momentum=quant_use_momentum, momentum=quant_momentum, percentile=quant_percentile, sample_ratio=quant_sample_ratio)
