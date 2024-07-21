import unittest

from scipy.stats import pearsonr

from torch import randn, no_grad
from torch.nn import Parameter, MSELoss, Conv2d
from torch.optim import SGD
from torch.testing import assert_close

from quantization.fixed_modules.parametric.convolution import QConv2D, QConv2DBatchNormReLU


class TestQuantizableConv2D(unittest.TestCase):
    def setUp(self):
        self.in_channels, self.out_channels, self.kernel_size = 25, 10, 3
        self.standard_conv2d = Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=False)
        self.input_tensor = randn(100, self.in_channels, 3, 3)
        # Quantization data type, the bit width should have a space of that can cover all the numbers
        # in_channels x out_channels. For example, 8 bits can cover the 2^8 = 256 numbers > 25 x 10 = 250.
        self.bit_width = [8, 8, 8]

    def _sync_weights(self, layer):
        with no_grad():
            layer.conv2d_module.weight = Parameter(self.standard_conv2d.weight.detach().clone())
        return layer

    def test_fp32_forward_pass_equality(self):
        quantizable_conv2d = QConv2D(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size,
                                     num_bits=self.bit_width,
                                     bias=False,
                                     qi=False,
                                     qo=False,
                                     )
        quantizable_conv2d = self._sync_weights(quantizable_conv2d)

        output_quantizable = quantizable_conv2d(self.input_tensor)
        output_standard = self.standard_conv2d(self.input_tensor)

        assert_close(output_quantizable.detach(), output_standard.detach(), atol=1e-5, rtol=1e-3)

    def test_fp32_weight_convergence(self):
        X, Y = randn(100, self.in_channels, 3, 3), randn(100, self.out_channels, 3, 3)
        criterion = MSELoss()
        optimizer_standard = SGD(self.standard_conv2d.parameters(), lr=0.01)

        quantizable_conv2d = QConv2D(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size,
                                     num_bits=self.bit_width,
                                     bias=False,
                                     qi=False,
                                     qo=False,
                                     )
        quantizable_conv2d = self._sync_weights(quantizable_conv2d)
        optimizer_quantizable = SGD(quantizable_conv2d.parameters(), lr=0.01)

        for _ in range(100):
            optimizer_standard.zero_grad()
            optimizer_quantizable.zero_grad()

            loss_standard = criterion(self.standard_conv2d(X), Y)
            loss_standard.backward()
            optimizer_standard.step()

            loss_quantizable = criterion(quantizable_conv2d(X), Y)
            loss_quantizable.backward()
            optimizer_quantizable.step()

        with no_grad():
            assert_close(quantizable_conv2d.conv2d_module.weight.data,
                         self.standard_conv2d.weight.data,
                         atol=1e-2,
                         rtol=1e-2)

    def test_int_weight_distribution(self):
        X = randn(100, self.in_channels, 3, 3)
        Y = self.standard_conv2d(X).detach()

        criterion = MSELoss()
        quantizable_conv2d = QConv2D(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size,
                                     num_bits=self.bit_width,
                                     bias=False,
                                     qi=True,
                                     qo=True,
                                     )
        quantizable_conv2d.forward = quantizable_conv2d.simulated_quantize_forward
        quantizable_conv2d = self._sync_weights(quantizable_conv2d)

        optimizer_quantizable = SGD(quantizable_conv2d.parameters(), lr=0.1)

        tolerance, max_epochs = 1e-3, 10000
        for i in range(max_epochs):
            optimizer_quantizable.zero_grad()
            loss = criterion(quantizable_conv2d(X), Y)
            if loss <= tolerance:
                break
            loss.backward()
            optimizer_quantizable.step()

        quantizable_conv2d.freeze()

        flatten_weight = self.standard_conv2d.weight.detach().clone().numpy().flatten()
        flatten_quantized_weight = quantizable_conv2d.quantized_weight.detach().clone().numpy().flatten()
        pearson_correlation, _ = pearsonr(flatten_weight, flatten_quantized_weight)
        self.assertGreaterEqual(pearson_correlation,
                                0.99,
                                msg=f"The correlation between the weight distributions is {pearson_correlation}, "
                                    f"with conversion tolerance of {tolerance} and {i} / {max_epochs} epochs.")

    def test_conv2d_batch_norm_relu(self):
        x = randn(100, self.in_channels, 3, 3)

        conv2d_bn_relu = QConv2DBatchNormReLU(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=self.kernel_size,
                                              bias=True,
                                              qi=True,
                                              qo=True,
                                              num_bits=self.bit_width,
                                              bn_affine=False,
                                              )
        y = conv2d_bn_relu(x)
        _ = conv2d_bn_relu.simulated_quantize_forward(x)
        conv2d_bn_relu.freeze()

        q_y = conv2d_bn_relu.qi.quantize(x)
        q_y = conv2d_bn_relu.quantize_inference(q_y)
        dq_y = conv2d_bn_relu.qo.dequantize(q_y)

        flatten_y = y.detach().clone().numpy().flatten()
        flatten_dq_y = dq_y.detach().clone().numpy().flatten()
        pearson_correlation, _ = pearsonr(flatten_y, flatten_dq_y)
        self.assertGreaterEqual(pearson_correlation,
                                0.9,
                                msg=f"The correlation between the weight distributions is {pearson_correlation}.")


if __name__ == '__main__':
    unittest.main()
