from typing import Literal

from torch import Tensor, tensor, float32
from torch.nn import Module, ModuleList, Parameter

from quantization.base_parameter import QuantizationParameters, MinMaxRangesQuantizationParameters


class MQModule(Module):
    def __init__(self, qi=False, qo=False, num_bits_list=None, is_signed=True, quantize_per=Literal['tensor', 'column', 'element']):
        super(MQModule, self).__init__()
        self.num_bits_list = num_bits_list
        if qi:
            self.qi = ModuleList([QuantizationParameters(num_bits=num_bits,
                                                         is_signed=is_signed,
                                                         quantize_per=quantize_per,
                                                         )
                                  for num_bits in num_bits_list])
            self.qi_relaxed_coeff = Parameter(Tensor(len(num_bits_list)))
        else:
            self.qi = None
        if qo:
            self.qo = ModuleList([QuantizationParameters(num_bits=num_bits,
                                                         is_signed=is_signed,
                                                         quantize_per=quantize_per,
                                                         )
                                  for num_bits in num_bits_list])
            self.qo_relaxed_coeff = Parameter(Tensor(len(num_bits_list)))
        else:
            self.qo = None

    def reset_parameters(self):
        if self.qi is not None:
            device = self.qi_relaxed_coeff.device
            [qi.reset_parameters() for qi in self.qi]
            self.qi_relaxed_coeff.data = tensor([bit for bit in self.num_bits_list], dtype=float32).softmax(0).softmax(0).data.to(device)
        if self.qo is not None:
            device = self.qo_relaxed_coeff.device
            [qo.reset_parameters() for qo in self.qo]
            self.qo_relaxed_coeff.data = tensor([bit for bit in self.num_bits_list], dtype=float32).softmax(0).softmax(0).data.to(device)


class MQNonParametricModule(MQModule):
    """
    A module that does not have any learnable parameters.
    Abstract class for non-parametric quantization modules, to be able to distinguish from non-parametric and parametric modules.
    """
    pass


class MQParametricModule(MQModule):
    """
    A module that has learnable parameters.
    Abstract class for parametric quantization modules, to be able to distinguish from non-parametric and parametric modules.
    """
    pass

