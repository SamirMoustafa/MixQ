import re
import warnings
from typing import Any

from math import sqrt
from torch import Tensor, tensor

from quantization.functional import define_quantization_ranges


def calculate_scale_and_zero_point(min_val, max_val, num_bits, signed, device="cpu"):
    qmin, qmax = define_quantization_ranges(num_bits, signed)

    scale = (max_val - min_val) / (qmax - qmin) + 1e-16
    zero_point = qmax - max_val / scale
    zero_point.round_()

    # if zero_point < qmin:
    #     zero_point = qmin
    # elif zero_point > qmax:
    #     zero_point = qmax

    return tensor(scale, device=device), tensor(zero_point, device=device)


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def camel_case_split(string, lower=True):
    """
    Convert camel case to snake case (e.g., CamelCase -> camel_case)

    :param string: String to convert
    :param lower: whether to convert to a lower case or not
    :return: converted string
    """
    string = " ".join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string))
    return string.lower() if lower else string


def _top_k_of_softmax_of_tensor(coeff_tensor, k):
    softmax_top_k = coeff_tensor.softmax(dim=0).topk(k, largest=True)
    # check if the top k values have similar values
    if len(softmax_top_k.values.unique()) != k:
        warnings.warn(f"The top {k} wining values {softmax_top_k.values} have similar values")
    return softmax_top_k.indices.tolist()
