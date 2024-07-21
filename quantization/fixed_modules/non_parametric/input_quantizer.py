from quantization.fixed_modules.base_module import QNonParametricModule


class QInput(QNonParametricModule):
    def __init__(self, qi=False, num_bits=None, is_signed=True, quantize_per="tensor"):
        super(QInput, self).__init__(qi=qi, qo=False, num_bits=[num_bits[0], -1], is_signed=is_signed, quantize_per=quantize_per)
        self.quantize_per = quantize_per

    def reset_parameters(self):
        super().reset_parameters()

    def freeze(self, qi=None):
        if self.quantize_per != 'tensor':
            raise NotImplementedError("Only tensor-wise quantization is supported for now.")
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        return x

    def simulated_quantize_forward(self, x):
        if self.qi is not None:
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)
        return x

    def quantize_inference(self, x):
        x = self.qi.quantize(x)
        return x
