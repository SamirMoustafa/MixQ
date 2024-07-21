from torch import tensor

from quantization.mixed_modules.base_module import MQNonParametricModule
from quantization.utility import _top_k_of_softmax_of_tensor


class MQInput(MQNonParametricModule):
    def __init__(self, qi=False, num_bits_list=None, is_signed=True, quantize_per="tensor"):
        super(MQInput, self).__init__(qi=qi, qo=False, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
        self.quantize_per = quantize_per
        self.num_bits_list = num_bits_list

    def reset_parameters(self):
        super().reset_parameters()

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

