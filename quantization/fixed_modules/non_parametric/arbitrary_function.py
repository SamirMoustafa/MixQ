from torch import linspace, tensor
from torch_operation_counter import OperationsCounterMode

from quantization.fixed_modules.base_module import QNonParametricModule
from quantization.functional import (define_quantization_ranges, torch_dequantize, torch_quantize,
                                     interp1d_linear, interp1d_nearest_neighbor)


class QNonParametricFunction1D(QNonParametricModule):

    def __init__(self, function, qi=True, qo=True, num_bits=None, lut_size=64, is_signed=True, quantize_per="tensor"):
        super(QNonParametricFunction1D, self).__init__(qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, quantize_per=quantize_per)
        assert function(tensor([0.])).shape == tensor([0.]).shape, "The function must be element-wise"
        self.function = function
        self.num_bits = num_bits
        self.lut_size = lut_size
        self.is_signed = is_signed
        self.quantize_per = quantize_per
        self.register_buffer('lut_qx', None)
        self.register_buffer('lut_qy', None)

    def forward(self, x):
        x = self.function(x)
        return x

    def simulated_quantize_forward(self, x):
        if self.qi is not None:
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)

        x = self.function(x)

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

        device = self.qi.scale.device
        qmin, qmax = define_quantization_ranges(self.num_bits, signed=self.is_signed)
        lut_qx = linspace(qmin, qmax, self.lut_size, device=device).round()

        lut_x = torch_dequantize(lut_qx, self.qi.scale, self.qi.zero_point)
        lut_y = self.function(lut_x)
        lut_qy = torch_quantize(lut_y, *self.qo.get_quantization_arguments())

        self.lut_qy = lut_qy
        self.lut_qx = lut_qx

    def quantize_inference(self, x):
        qmin, qmax = define_quantization_ranges(self.num_bits, signed=self.is_signed)
        x = interp1d_linear(x, self.lut_qx, self.lut_qy)
        x = x.round().clamp(qmin, qmax)
        return x

    def estimated_bit_operation_precision(self, x):
        bit_widths = []
        if self.qi is not None:
            bit_widths += [self.qi.num_bits]
        if self.qo is not None:
            bit_widths += [self.qo.num_bits]
        expected_bit_width = sum(bit_widths) / len(bit_widths) if len(bit_widths) > 0 else 0
        with OperationsCounterMode(self) as ops_counter:
            self.function(x)
        return ops_counter.total_main_operation * expected_bit_width

if __name__ == "__main__":
    import inspect
    from torch import linspace, norm
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    function_f = lambda x: F.sigmoid(x)

    # Parameters for plotting
    x_values = linspace(-10, 10, 500)  # Range for the function
    original_values = function_f(x_values)  # Original sine values

    # Instantiate the quantizable function module
    num_bits = 4
    quantized_module = QNonParametricFunction1D(function_f, num_bits=num_bits, is_signed=False)
    _ = quantized_module.simulated_quantize_forward(x_values)  # Calibrate the quantization parameters
    quantized_module.freeze()  # Prepare the module for quantized inference
    quantized_values = quantized_module.qi.quantize(x_values)
    quantized_values = quantized_module.quantize_inference(quantized_values)
    quantized_values = quantized_module.qo.dequantize(quantized_values)

    quantization_error_l2 = round(norm(quantized_values - original_values, p=2).item(), 3)
    # Plot the original and quantized functions
    function_name = inspect.getsource(function_f).split(':')[1].strip().split('(')[0]
    plt.figure(figsize=(10, 6))
    plt.plot(x_values.numpy(), original_values.numpy(), label=f"Original -- {function_name} Function")
    plt.plot(x_values.detach().numpy(), quantized_values.detach().numpy(), label=f"INT-{num_bits} -- {function_name} Function")
    plt.fill_between(x_values.numpy(), original_values.numpy(), quantized_values.detach().numpy(), color='tomato', alpha=0.5, label="$||\\text{Quantization Error}||_2=$" + str(quantization_error_l2))

    plt.xlabel(f"x")
    plt.ylabel(f"{function_name}(x)")
    plt.legend()
    plt.show()