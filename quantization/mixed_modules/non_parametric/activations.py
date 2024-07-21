from torch import tensor
from torch.nn.functional import relu, sigmoid

from torch_operation_counter import OperationsCounterMode

from quantization.mixed_modules.base_module import MQNonParametricModule
from quantization.mixed_modules.non_parametric.arbitrary_function import MQNonParametricFunction1D
from quantization.utility import _top_k_of_softmax_of_tensor
from utility import flatten_list


class MQReLU(MQNonParametricModule):
    def __init__(self, qi=False, num_bits_list=None, quantize_per="tensor"):
        super(MQReLU, self).__init__(qi=qi, qo=False, num_bits_list=num_bits_list, is_signed=False, quantize_per=quantize_per)
        self.quantize_per = quantize_per
        self.num_bits_list = num_bits_list

    def relaxed_quantize_input(self, x, q_coeff_softmax):
        if self.qi is not None:
            xs = []
            for i, q in enumerate(self.qi):
                q.calibrate(x)
                xs += [q.fake_quantize(x) * q_coeff_softmax[i]]
            x = sum(xs)
        return x

    def forward(self, x):
        if self.qi is not None:
            qi_coeff_softmax = self.qi_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_input(x, qi_coeff_softmax)
        x = relu(x)
        return x

    def calculate_weighted_loss(self, x):
        device = x.device
        loss = tensor([0.0], device=device)
        bits_tensor = tensor([self.num_bits_list], device=device, dtype=x.dtype)

        if self.qi is not None:
            qi_coeff_softmax = self.qi_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_input(x, qi_coeff_softmax)
            loss += (bits_tensor * qi_coeff_softmax).sum() * x.numel()

        return loss / (1024 * 8)

    def select_top_k_winners(self, k):
        wining_bit_width = {"qi": []}
        if self.qi is not None:
            top_k_qi = _top_k_of_softmax_of_tensor(self.qi_relaxed_coeff, k)
            wining_bit_width["qi"] = [self.num_bits_list[i] for i in top_k_qi]
        else:
            wining_bit_width["qi"] = [None for _ in range(k)]
        return wining_bit_width

    def estimated_bit_operation_precision(self, x):
        wining_bit_width = [*filter(None, flatten_list([*self.select_top_k_winners(1).values()]))]
        if len(wining_bit_width) > 0:
            with OperationsCounterMode(self) as ops_counter:
                relu(x)
            expected_bit_width = sum(wining_bit_width) / len(wining_bit_width)
            return ops_counter.total_main_operation * expected_bit_width
        else:
            return 0


class MQSigmoid(MQNonParametricFunction1D):

    def __init__(self, qi=True, qo=True, num_bits_list=None, lut_size=64):
        super(MQSigmoid, self).__init__(function=sigmoid,
                                        qi=qi,
                                        qo=qo,
                                        num_bits_list=num_bits_list,
                                        lut_size=lut_size,
                                        is_signed=False)
