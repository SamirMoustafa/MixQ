import unittest

from scipy.stats import pearsonr

from torch import randn, no_grad
from torch.nn import Parameter, MSELoss, Linear
from torch.optim import SGD
from torch.testing import assert_close

from quantization.fixed_modules.parametric.linear import QLinear, QLinearBatchNormReLU


class TestQuantizableLinear(unittest.TestCase):
    def setUp(self):
        self.in_channels, self.out_channels = 25, 10
        self.standard_linear = Linear(self.in_channels, self.out_channels, bias=False)
        self.input_tensor = randn((1, self.in_channels))
        # Quantization data type, the bit width should have a space of that can cover all the numbers
        # in_channels x out_channels. For example, 8 bits can cover the 2^8 = 256 numbers > 25 x 10 = 250.
        self.bit_width = [8, 8, 8]

    def _sync_weights(self, layer):
        with no_grad():
            layer.linear_module.weight = Parameter(self.standard_linear.weight.detach().clone())
        return layer

    def test_fp32_forward_pass_equality(self):
        quantizable_linear = QLinear(in_features=self.in_channels,
                                     out_features=self.out_channels,
                                     num_bits=self.bit_width,
                                     bias=False,
                                     qi=False,
                                     qo=False,
                                     )
        quantizable_linear = self._sync_weights(quantizable_linear)

        output_quantizable = quantizable_linear(self.input_tensor)
        output_standard = self.standard_linear(self.input_tensor)

        assert_close(output_quantizable.detach(), output_standard.detach(), atol=1e-5, rtol=1e-3)

    def test_fp32_weight_convergence(self):
        X, Y = randn(100, self.in_channels), randn(100, self.out_channels)
        criterion = MSELoss()
        optimizer_standard = SGD(self.standard_linear.parameters(), lr=0.01)

        quantizable_linear = QLinear(in_features=self.in_channels,
                                     out_features=self.out_channels,
                                     num_bits=self.bit_width,
                                     bias=False,
                                     qi=False,
                                     qo=False,
                                     )
        quantizable_linear = self._sync_weights(quantizable_linear)
        optimizer_quantizable = SGD(quantizable_linear.parameters(), lr=0.01)

        for _ in range(100):
            optimizer_standard.zero_grad()
            optimizer_quantizable.zero_grad()

            loss_standard = criterion(self.standard_linear(X), Y)
            loss_standard.backward()
            optimizer_standard.step()

            loss_quantizable = criterion(quantizable_linear(X), Y)
            loss_quantizable.backward()
            optimizer_quantizable.step()

        with no_grad():
            assert_close(quantizable_linear.linear_module.weight.data,
                         self.standard_linear.weight.data,
                         atol=1e-2,
                         rtol=1e-2)

    def test_int_weight_distribution(self):
        X = randn(100, self.in_channels)
        Y = self.standard_linear(X).detach()

        criterion = MSELoss()
        quantizable_linear = QLinear(in_features=self.in_channels,
                                     out_features=self.out_channels,
                                     num_bits=self.bit_width,
                                     bias=False,
                                     qi=True,
                                     qo=True,
                                     )
        quantizable_linear.forward = quantizable_linear.simulated_quantize_forward
        quantizable_linear = self._sync_weights(quantizable_linear)

        optimizer_quantizable = SGD(quantizable_linear.parameters(), lr=0.1)

        tolerance, max_epochs = 1e-3, 10000
        for i in range(max_epochs):
            optimizer_quantizable.zero_grad()
            loss = criterion(quantizable_linear(X), Y)
            if loss <= tolerance:
                break
            loss.backward()
            optimizer_quantizable.step()

        quantizable_linear.freeze()

        flatten_weight = self.standard_linear.weight.detach().clone().numpy().flatten()
        flatten_quantized_weight = quantizable_linear.quantized_weight.detach().clone().numpy().flatten()
        pearson_correlation, _ = pearsonr(flatten_weight, flatten_quantized_weight)
        self.assertGreaterEqual(pearson_correlation,
                                0.99,
                                msg=f"The correlation between the weight distributions is {pearson_correlation}, "
                                    f"with conversion tolerance of {tolerance} and {i} / {max_epochs} epochs.")

    def test_linear_batch_norm_relu(self):
        x = randn(100, self.in_channels)

        linear_bn_relu = QLinearBatchNormReLU(in_features=self.in_channels,
                                              out_features=self.out_channels,
                                              bias=True,
                                              qi=True,
                                              qo=True,
                                              num_bits=self.bit_width,
                                              bn_affine=False,
                                              )
        y = linear_bn_relu(x)
        _ = linear_bn_relu.simulated_quantize_forward(x)
        linear_bn_relu.freeze()

        q_y = linear_bn_relu.qi.quantize(x)
        q_y = linear_bn_relu.quantize_inference(q_y)
        dq_y = linear_bn_relu.qo.dequantize(q_y)

        flatten_y = y.detach().clone().numpy().flatten()
        flatten_dq_y = dq_y.detach().clone().numpy().flatten()
        pearson_correlation, _ = pearsonr(flatten_y, flatten_dq_y)
        self.assertGreaterEqual(pearson_correlation,
                                0.9,
                                msg=f"The correlation between the weight distributions is {pearson_correlation}.")


if __name__ == '__main__':
    unittest.main()
