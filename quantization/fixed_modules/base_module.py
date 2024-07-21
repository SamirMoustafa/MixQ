from typing import Literal

from torch.nn import Module

from quantization.base_parameter import QuantizationParameters, MinMaxRangesQuantizationParameters


class QModule(Module):
    def __init__(self, qi=False, qo=False, num_bits=None, is_signed=True, quantize_per=Literal['tensor', 'column', 'element']):
        super(QModule, self).__init__()
        self.qi = QuantizationParameters(num_bits=num_bits[0], is_signed=is_signed, quantize_per=quantize_per) if qi else None
        self.qo = QuantizationParameters(num_bits=num_bits[1], is_signed=is_signed, quantize_per=quantize_per) if qo else None

    def reset_parameters(self):
        if self.qi is not None:
            self.qi.reset_parameters()
        if self.qo is not None:
            self.qo.reset_parameters()

    def freeze(self, qi=None, qo=None):
        if self.qi is not None and qi is not None:
            raise ValueError("qi has been provided in init function.")
        if self.qi is None and qi is None:
            raise ValueError("qi is not existed, should be provided.")

        if self.qo is not None and qo is not None:
            raise ValueError("qo has been provided in init function.")
        if self.qo is None and qo is None:
            raise ValueError("qo is not existed, should be provided.")

    def quantize_inference(self, x):
        raise NotImplementedError("quantize_inference should be implemented.")


class QMinMaxRangesModule(QModule):
    def __init__(self, qi=False, qo=False, num_bits=None, is_signed=True, use_momentum=True, momentum: float = 0.0, percentile: float = 1.0, sample_ratio: float = 1.0, quantize_per=Literal['tensor', 'column', 'element']):
        super(QMinMaxRangesModule, self).__init__(qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, quantize_per=quantize_per)
        self.qi = MinMaxRangesQuantizationParameters(num_bits=num_bits[0], is_signed=is_signed, use_momentum=use_momentum, momentum=momentum, percentile=percentile, sample_ratio=sample_ratio) if qi else None
        self.qo = MinMaxRangesQuantizationParameters(num_bits=num_bits[1], is_signed=is_signed, use_momentum=use_momentum, momentum=momentum, percentile=percentile, sample_ratio=sample_ratio) if qo else None


class QNonParametricModule(QModule):
    """
    A module that does not have any learnable parameters.
    Abstract class for non-parametric quantization modules, to be able to distinguish from non-parametric and parametric modules.
    """
    pass


class QMinMaxRangesNonParametricModule(QMinMaxRangesModule):
    """
    A module that does not have any learnable parameters.
    Abstract class for non-parametric quantization modules, to be able to distinguish from non-parametric and parametric modules.
    """
    pass


class QParametricModule(QModule):
    """
    A module that has learnable parameters.
    Abstract class for parametric quantization modules, to be able to distinguish from non-parametric and parametric modules.
    """
    pass


class QMinMaxRangesParametricModule(QMinMaxRangesModule):
    """
    A module that has learnable parameters.
    Abstract class for parametric quantization modules, to be able to distinguish from non-parametric and parametric modules.
    """
    pass
