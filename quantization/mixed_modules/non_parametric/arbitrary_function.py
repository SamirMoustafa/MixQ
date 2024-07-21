from torch import tensor

from torch_operation_counter import OperationsCounterMode

from quantization.mixed_modules.base_module import MQNonParametricModule
from quantization.utility import _top_k_of_softmax_of_tensor
from utility import flatten_list


class MQNonParametricFunction1D(MQNonParametricModule):

    def __init__(self, function, qi=True, qo=True, num_bits_list=None, lut_size=64, is_signed=True, quantize_per="tensor"):
        super(MQNonParametricFunction1D, self).__init__(qi=qi, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
        assert function(tensor([0.])).shape == tensor([0.]).shape, "The function must be element-wise"
        self.function = function
        self.num_bits_list = num_bits_list
        self.lut_size = lut_size
        self.is_signed = is_signed
        self.quantize_per = quantize_per
        self.register_buffer('lut_qx', None)
        self.register_buffer('lut_qy', None)

    def reset_parameters(self):
        super().reset_parameters()
        self.lut_qx = None
        self.lut_qy = None

    def relaxed_quantize_input(self, x, q_coeff_softmax):
        if self.qi is not None:
            xs = []
            for i, q in enumerate(self.qi):
                q.calibrate(x)
                xs += [q.fake_quantize(x) * q_coeff_softmax[i]]
            x = sum(xs)
        return x

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

        x = self.function(x)

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

        x = self.function(x)

        if self.qo is not None:
            qo_coeff_softmax = self.qo_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_output(x, qo_coeff_softmax)
            loss += (bits_tensor * qo_coeff_softmax).sum() * x.numel()

        return loss / (1024 * 8)

    def select_top_k_winners(self, k):
        wining_bit_width = {"qi": [],  "qo": []}
        if self.qi is not None:
            top_k_qi = _top_k_of_softmax_of_tensor(self.qi_relaxed_coeff, k)
            wining_bit_width["qi"] = [self.num_bits_list[i] for i in top_k_qi]
        else:
            wining_bit_width["qi"] = [None for _ in range(k)]

        if self.qo is not None:
            top_k_qo = _top_k_of_softmax_of_tensor(self.qo_relaxed_coeff, k)
            wining_bit_width["qo"] = [self.num_bits_list[i] for i in top_k_qo]
        else:
            wining_bit_width["qo"] = [None for _ in range(k)]
        return wining_bit_width

    def estimated_bit_operation_precision(self, x):
        wining_bit_width = [*filter(None, flatten_list([*self.select_top_k_winners(1).values()]))]
        if len(wining_bit_width) > 0:
            with OperationsCounterMode(self) as ops_counter:
                self.function(x)
            expected_bit_width = sum(wining_bit_width) / len(wining_bit_width)
            return ops_counter.total_main_operation * expected_bit_width
        else:
            return 0
