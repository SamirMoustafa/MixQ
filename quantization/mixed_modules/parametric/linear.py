from torch import Tensor, tensor, float32
from torch.nn import Linear, Parameter, ModuleList
from torch.nn.functional import linear

from torch_operation_counter import OperationsCounterMode

from quantization.mixed_modules.base_module import MQParametricModule
from quantization.base_parameter import QuantizationParameters
from quantization.utility import _top_k_of_softmax_of_tensor
from utility import flatten_list


class MQLinear(MQParametricModule):
    def __init__(self,
                 in_features,
                 out_features,
                 qi,
                 qo,
                 num_bits_list,
                 bias=True,
                 is_signed=False,
                 quantize_per="tensor",
                 ):
        super().__init__(qi=qi, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
        self.in_features, self.out_features = in_features, out_features
        self.linear_modules = ModuleList([Linear(in_features, out_features, bias=bias) for _ in num_bits_list])
        self.qw = ModuleList([QuantizationParameters(num_bits=num_bits, is_signed=True, quantize_per=quantize_per)
                              for num_bits in num_bits_list])
        self.qw_relaxed_coeff = Parameter(Tensor(len(num_bits_list)))
        self.num_bits_list = num_bits_list
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for linear_module in self.linear_modules:
            linear_module.reset_parameters()
        for qi in self.qw:
            qi.reset_parameters()
        device = self.qw_relaxed_coeff.device
        self.qw_relaxed_coeff.data = tensor([bit for bit in self.num_bits_list], dtype=float32).softmax(0).softmax(0).data.to(device)
        self.synchronize_all_linear_weights_with_linear_0()

    def synchronize_all_linear_weights_with_linear_0(self):
        for i in range(1, len(self.linear_modules)):
            self.linear_modules[i].weight.data = self.linear_modules[0].weight.data
            if self.linear_modules[i].bias is not None:
                self.linear_modules[i].bias.data = self.linear_modules[0].bias.data

    def relaxed_quantize_input(self, x, q_coeff_softmax):
        if self.qi is not None:
            xs = []
            for i, q in enumerate(self.qi):
                q.calibrate(x)
                xs += [q.fake_quantize(x) * q_coeff_softmax[i]]
            x = sum(xs)
        return x

    def relaxed_quantize_weights(self, q_coeff_softmax):
        weights = [linear_module.weight for linear_module in self.linear_modules]
        quantized_weights = []
        for i, (qw, weight) in enumerate(zip(self.qw, weights)):
            qw.calibrate(weight)
            quantized_weights += [qw.fake_quantize(weight) * q_coeff_softmax[i]]
        return sum(quantized_weights)

    def relaxed_quantize_output(self, x, q_coeff_softmax):
        if self.qo is not None:
            xs = []
            for i, q in enumerate(self.qo):
                q.calibrate(x)
                xs += [q.fake_quantize(x) * q_coeff_softmax[i]]
            x = sum(xs)
        return x

    def forward(self, x):
        if self.qi is not None:
            qi_coeff_softmax = self.qi_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_input(x, qi_coeff_softmax)

        qw_coeff_softmax = self.qw_relaxed_coeff.softmax(dim=0)
        weight = self.relaxed_quantize_weights(qw_coeff_softmax)

        x = linear(x, weight)

        if self.qo is not None:
            qo_coeff_softmax = self.qo_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_output(x, qo_coeff_softmax)

        return x

    def calculate_weighted_loss(self, x):
        device = x.device
        loss = tensor([0.0], device=device)
        bits_tensor = tensor([self.num_bits_list], device=device, dtype=x.dtype)

        if self.qi is not None:
            qi_coeff_softmax = self.qi_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_input(x, qi_coeff_softmax)
            loss += (bits_tensor * qi_coeff_softmax).sum() * x.numel()

        qw_coeff_softmax = self.qw_relaxed_coeff.softmax(dim=0)
        weight = self.relaxed_quantize_weights(qw_coeff_softmax)
        loss += (bits_tensor * qw_coeff_softmax).sum() * weight.numel()

        x = linear(x, weight)

        if self.qo is not None:
            qo_coeff_softmax = self.qo_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_output(x, qo_coeff_softmax)
            loss += (bits_tensor * qo_coeff_softmax).sum() * x.numel()

        return loss / (1024 * 8)

    def select_top_k_winners(self, k):
        wining_bit_width = {"qi": [], "qw": [], "qo": []}
        if self.qi is not None:
            top_k_qi = _top_k_of_softmax_of_tensor(self.qi_relaxed_coeff, k)
            wining_bit_width["qi"] = [self.num_bits_list[i] for i in top_k_qi]
        else:
            wining_bit_width["qi"] = [None for _ in range(k)]

        top_k_qw = _top_k_of_softmax_of_tensor(self.qw_relaxed_coeff, k)
        wining_bit_width["qw"] = [self.num_bits_list[i] for i in top_k_qw]

        if self.qo is not None:
            top_k_qo = _top_k_of_softmax_of_tensor(self.qo_relaxed_coeff, k)
            wining_bit_width["qo"] = [self.num_bits_list[i] for i in top_k_qo]
        else:
            wining_bit_width["qo"] = [None for _ in range(k)]
        return wining_bit_width

    def estimated_bit_operation_precision(self, x):
        wining_bit_width = [*filter(None, flatten_list([*self.select_top_k_winners(1).values()]))]
        expected_bit_width = sum(wining_bit_width) / len(wining_bit_width)
        with OperationsCounterMode(self) as ops_counter:
            self.linear_modules[0].forward(x)
        return ops_counter.total_main_operation * expected_bit_width
