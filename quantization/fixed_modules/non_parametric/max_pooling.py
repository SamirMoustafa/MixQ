from torch.nn.functional import max_pool2d

from quantization.fixed_modules.base_module import QNonParametricModule


class QMaxPooling2D(QNonParametricModule):
    def __init__(self, kernel_size=3, stride=1, padding=0, qi=False, num_bits=None, is_signed=True, quantize_per="tensor"):
        super(QMaxPooling2D, self).__init__(qi=qi, qo=False, num_bits=[num_bits[0], -1], is_signed=is_signed, quantize_per=quantize_per)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.quantize_per = quantize_per

    def reset_parameters(self):
        super().reset_parameters()

    def freeze(self, qi=None):
        if self.quantize_per != 'tensor':
            raise NotImplementedError("Only tensor-wise quantization is supported for now.")
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        x = max_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x

    def simulated_quantize_forward(self, x):
        if self.qi is not None:
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)
        x = self(x)
        return x

    def quantize_inference(self, x):
        return self(x)
