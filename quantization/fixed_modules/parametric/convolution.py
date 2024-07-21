from torch import tensor, sqrt
from torch.autograd import Variable
from torch.nn import Conv2d, BatchNorm2d
from torch.nn.functional import relu, conv2d

from quantization.fixed_modules.base_module import QParametricModule
from quantization.base_parameter import QuantizationParameters
from quantization.functional import torch_quantize, define_quantization_ranges


class QConv2D(QParametricModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 qi,
                 qo,
                 num_bits,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 is_signed=False,
                 quantize_per='tensor',
                 ):
        super(QConv2D, self).__init__(qi=qi, qo=qo, num_bits=num_bits[:2], is_signed=is_signed, quantize_per=quantize_per)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding, self.dilation, self.groups = kernel_size, stride, padding, dilation, groups
        self.conv2d_module = Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=bias,
                                    )
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
        self.M.data = tensor([], requires_grad=False)
        self.conv2d_module.reset_parameters()

    def freeze(self, qi=None, qo=None):
        if self.quantize_per != 'tensor':
            raise NotImplementedError("Only tensor-wise quantization is supported for now.")
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = self.qw.scale * self.qi.scale / self.qo.scale

        self.quantized_weight = self.qw.quantize(self.conv2d_module.weight)
        self.quantized_weight = (self.quantized_weight - self.qw.zero_point).round()

        if self.conv2d_module.bias is not None:
            fused_scale = self.qi.scale * self.qw.scale
            qmin, qmax = define_quantization_ranges(32, signed=True)
            self.quantized_bias.data = torch_quantize(self.conv2d_module.bias, scale=fused_scale, zero_point=0, qmin=qmin, qmax=qmax)

    def forward(self, x):
        x = self.conv2d_module(x)
        return x

    def simulated_quantize_forward(self, x):
        if self.qi is not None:
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)

        weight, bias = self.conv2d_module.weight, self.conv2d_module.bias
        self.qw.calibrate(weight)
        weight = self.qw.fake_quantize(weight)
        self.conv2d_module.weight.data = weight
        if self.conv2d_module.bias is not None:
            self.conv2d_module.bias.data = bias.data

        x = self.conv2d_module(x)

        if self.qo is not None:
            self.qo.calibrate(x)
            x = self.qo.fake_quantize(x)

        return x

    def quantize_inference(self, x):
        qmin, qmax = define_quantization_ranges(self.num_bits[1], signed=self.is_signed)
        x = x - self.qi.zero_point
        x = conv2d(x,
                   self.quantized_weight,
                   self.quantized_bias,
                   stride=self.conv2d_module.stride,
                   padding=self.conv2d_module.padding,
                   dilation=self.conv2d_module.dilation,
                   groups=self.conv2d_module.groups,
                   )
        x = self.M * x
        x = x.round()
        x = x + self.qo.zero_point
        x = x.clamp(qmin, qmax)
        return x


class QConv2DBatchNormReLU(QParametricModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 qi,
                 qo,
                 num_bits,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 bn_affine=True,
                 eps=1e-5,
                 momentum=0.1,
                 is_signed=False,
                 quantize_per="tensor",
                 ):
        super(QConv2DBatchNormReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits[:2], is_signed=is_signed, quantize_per=quantize_per)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding, self.dilation, self.groups = kernel_size, stride, padding, dilation, groups
        self.conv2d_module = Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=bias,
                                    )
        self.batch_norm_module = BatchNorm2d(num_features=out_channels, eps=eps, momentum=momentum, affine=bn_affine)

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
        self.M.data = tensor([], requires_grad=False)
        self.conv2d_module.reset_parameters()
        self.batch_norm_module.reset_parameters()

    def fold_bn(self, mean, std):
        if self.batch_norm_module.affine:
            gamma_ = self.batch_norm_module.weight / std
            weight = self.conv2d_module.weight * gamma_.view(self.out_channels, 1, 1, 1)
            if self.conv2d_module.bias is not None:
                bias = gamma_ * self.conv2d_module.bias - gamma_ * mean + self.batch_norm_module.bias
            else:
                bias = self.batch_norm_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv2d_module.weight * gamma_.view(self.out_channels, 1, 1, 1)
            if self.conv2d_module.bias is not None:
                bias = gamma_ * self.conv2d_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
        return weight, bias

    def forward(self, x):
        x = self.conv2d_module(x)
        x = self.batch_norm_module(x)
        x = relu(x)
        return x

    def simulated_quantize_forward(self, x):
        if self.qi is not None:
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)

        if self.training:
            y = self.conv2d_module(x)
            y = y.permute(1, 0, 2, 3)
            y = y.contiguous().view(self.out_channels, -1)
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

        x = conv2d(x,
                   weight,
                   bias,
                   stride=self.stride,
                   padding=self.padding,
                   dilation=self.dilation,
                   groups=self.groups,
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

        fused_scale = self.qi.scale * self.qw.scale
        qmin, qmax = define_quantization_ranges(32, signed=True)
        self.quantized_bias = torch_quantize(bias, scale=fused_scale, zero_point=0, qmin=qmin, qmax=qmax)

    def quantize_inference(self, x):
        qmin, qmax = define_quantization_ranges(self.num_bits[1], signed=self.is_signed)
        x = x - self.qi.zero_point
        x = conv2d(x,
                   self.quantized_weight,
                   self.quantized_bias,
                   stride=self.stride,
                   padding=self.padding,
                   dilation=self.dilation,
                   groups=self.groups,
                   )
        x = self.M * x
        x = x.round()
        x = x + self.qo.zero_point
        x = x.clamp(qmin, qmax)
        return x
