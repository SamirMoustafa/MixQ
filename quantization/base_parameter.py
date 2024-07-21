from typing import Literal

from torch import tensor, kthvalue, min as torch_min, max as torch_max
from torch.nn import Module, Parameter

from quantization.functional import (define_quantization_ranges, torch_quantize, torch_dequantize, torch_fake_quantize,
                                     sample_tensor)
from quantization.utility import calculate_scale_and_zero_point


class QuantizationParameters(Module):
    def __init__(self, num_bits: int, is_signed: bool, quantize_per=Literal['tensor', 'column', 'element']):
        super(QuantizationParameters, self).__init__()
        self.quantize_per = quantize_per
        self.num_bits = num_bits
        self.is_signed = is_signed
        self.arguments = {"num_bits": num_bits, "is_signed": is_signed}

        self.register_parameter("scale", Parameter(tensor([], requires_grad=True)))
        self.register_parameter("zero_point", Parameter(tensor([], requires_grad=True)))
        self.register_buffer("min_data_range", tensor([], requires_grad=False))
        self.register_buffer("max_data_range", tensor([], requires_grad=False))
        self.is_calibrated = False

        self.reset_parameters()

    def reset_parameters(self):
        self.scale = Parameter(tensor([], requires_grad=True))
        self.zero_point = Parameter(tensor([], requires_grad=True))
        self.min_data_range.data = tensor([], requires_grad=False)
        self.max_data_range.data = tensor([], requires_grad=False)
        self.is_calibrated = False

    def calibrate(self, x):
        if self.is_calibrated:
            return
        if self.max_data_range.nelement() == 0 or self.max_data_range.data < x.max().data:
            if self.quantize_per == "tensor":
                self.max_data_range.data = x.max().data
            elif self.quantize_per == "column":
                self.max_data_range.data = x.max(dim=0).values
            elif self.quantize_per == "element":
                raise NotImplementedError("Element-wise quantization is not supported yet.")
        self.max_data_range.clamp_(min=0)

        if self.min_data_range.nelement() == 0 or self.min_data_range.data > x.min().data:
            if self.quantize_per == "tensor":
                self.min_data_range.data = x.min().data
            elif self.quantize_per == "column":
                self.min_data_range.data = x.min(dim=0).values
            elif self.quantize_per == "element":
                raise NotImplementedError("Element-wise quantization is not supported yet.")
        self.min_data_range.clamp_(max=0)

        device = x.device
        scale, zero_point = calculate_scale_and_zero_point(self.min_data_range, self.max_data_range, self.num_bits, self.is_signed, device=device)
        self.scale.data = tensor(scale.tolist(), device=device, requires_grad=True)
        self.zero_point.data = tensor(zero_point.tolist(), device=device, requires_grad=True)
        self.is_calibrated = True

    def get_quantization_arguments(self):
        qmin, qmax = define_quantization_ranges(self.num_bits, signed=self.is_signed)
        return self.scale, self.zero_point, qmin, qmax

    def quantize(self, x):
        qmin, qmax = define_quantization_ranges(self.num_bits, signed=self.is_signed)
        return torch_quantize(x, self.scale, self.zero_point, qmin, qmax)

    def dequantize(self, q_x):
        return torch_dequantize(q_x, self.scale, self.zero_point)

    def fake_quantize(self, x):
        qmin, qmax = define_quantization_ranges(self.num_bits, signed=self.is_signed)
        return torch_fake_quantize(x, self.scale, self.zero_point, qmin, qmax)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key_names = ["scale", "zero_point", "min_data_range", "max_data_range"]
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __repr__(self):
        return "{}(num_bits={}, scale={}, zero_point={}, min={}, max={})".format(self.__class__.__name__,
                                                                                 self.num_bits,
                                                                                 self.scale.shape,
                                                                                 self.zero_point.shape,
                                                                                 self.min_data_range.shape,
                                                                                 self.max_data_range.shape,
                                                                                 )

    def copy(self):
        cloned_instance = self.__class__(**self.arguments)
        cloned_instance.scale.data = self.scale.data.clone()
        cloned_instance.zero_point.data = self.zero_point.data.clone()
        cloned_instance.min_data_range = self.min_data_range.clone()
        cloned_instance.max_data_range = self.max_data_range.clone()
        cloned_instance.is_calibrated = self.is_calibrated
        return cloned_instance


class MinMaxRangesQuantizationParameters(QuantizationParameters):
    # Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, 2020
    def __init__(self, num_bits: int, is_signed: bool, use_momentum: bool, momentum: float, percentile: float, sample_ratio: float):
        super(MinMaxRangesQuantizationParameters, self).__init__(num_bits, is_signed)
        self.momentum_min_max = use_momentum
        self.momentum = momentum
        self.percentile = percentile
        self.sample_ratio = sample_ratio
        self.arguments.update({"use_momentum": use_momentum, "momentum": momentum, "percentile": percentile, "sample_ratio": sample_ratio})

    def compute_min_max(self, t, percentile):
        flat_t = t.flatten()
        numel = t.numel()
        if percentile is not None:
            k_min = max(1, int(numel * percentile))
            k_max = min(numel, max(1, int(numel * (1 - percentile))))
            min_val = kthvalue(flat_t, k_min)[0]
            max_val = kthvalue(flat_t, k_max)[0]
        else:
            min_val = torch_min(flat_t)
            max_val = torch_max(flat_t)
        return min_val, max_val

    def update_data_ranges(self, x):
        if self.sample_ratio is not None:
            x = sample_tensor(self.sample_ratio, x)
        current_min, current_max = self.compute_min_max(x, self.percentile)

        if self.min_data_range.numel() == 0 or self.max_data_range.numel() == 0:
            self.min_data_range, self.max_data_range = current_min, current_max
        else:
            if self.momentum_min_max:
                self.min_data_range += self.momentum * (current_min - self.min_data_range)
                self.max_data_range += self.momentum * (current_max - self.max_data_range)
            else:
                self.min_data_range = torch_min(current_min, self.min_data_range)
                self.max_data_range = torch_max(current_max, self.max_data_range)

        self.is_calibrated = True

    def calibrate(self, x):
        # In Degree-Quant original implementation, the calibration is done every time the quantization is performed.
        # However, in this implementation, the calibration is done only once, and allows the gradients to update quantization parameters.
        # To have the same behavior as the original implementation, comment on the following two lines.
        if self.is_calibrated:
            return
        if self.training:
            self.update_data_ranges(x.detach())

        device = self.scale.device
        scale, zero_point = calculate_scale_and_zero_point(self.min_data_range, self.max_data_range, self.num_bits, self.is_signed, device=device)
        self.scale.data = tensor(scale.tolist(), device=device, requires_grad=True)
        self.zero_point.data = tensor(zero_point.tolist(), device=device, requires_grad=True)
