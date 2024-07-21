from torch.nn.functional import relu, sigmoid
from torch_operation_counter import OperationsCounterMode

from quantization.fixed_modules.base_module import QNonParametricModule
from quantization.fixed_modules.non_parametric.arbitrary_function import QNonParametricFunction1D


class QReLU(QNonParametricModule):
    def __init__(self, qi=False, num_bits=None, quantize_per="tensor"):
        super(QReLU, self).__init__(qi=qi, qo=False, num_bits=[num_bits[0], -1], is_signed=False, quantize_per=quantize_per)
        self.quantize_per = quantize_per

    def reset_parameters(self):
        if self.qi is not None:
            self.qi.reset_parameters()

    def freeze(self, qi=None):
        if self.quantize_per != 'tensor':
            raise NotImplementedError("Only tensor-wise quantization is supported for now.")
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        x = relu(x)
        return x

    def simulated_quantize_forward(self, x):
        if self.qi is not None:
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)
        x = relu(x)
        return x

    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x

    def estimated_bit_operation_precision(self, x):
        if self.qi is not None:
            bit_widths = self.qi.num_bits
            with OperationsCounterMode(self) as ops_counter:
                relu(x)
            return ops_counter.total_main_operation * bit_widths
        else:
            return 0


class QSigmoid(QNonParametricFunction1D):

    def __init__(self, qi=True, qo=True, num_bits=None, lut_size=64):
        super(QSigmoid, self).__init__(function=sigmoid,
                                       qi=qi,
                                       qo=qo, num_bits=num_bits[0],
                                       lut_size=lut_size,
                                       is_signed=False)
