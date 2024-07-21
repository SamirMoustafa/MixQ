import inspect
from typing import Any
from collections import OrderedDict

from numpy import ndarray

from torch import is_tensor

msg_special_args = {"edge_index", "edge_index_i", "edge_index_j", "size", "size_i", "size_j"}

aggr_special_args = {"index", "dim_size"}


def __process_size__(size):
    size = [None, None] if size is None else size
    size = [size, size] if isinstance(size, int) else size
    size = size.tolist() if is_tensor(size) else size
    size = list(size) if isinstance(size, tuple) else size
    assert isinstance(size, list), f"Size must be a list, but is of type {type(size)}."
    assert len(size) == 2, f"Size list must be of length 2, but is of length {len(size)}."
    return size


def __single_or_tuple__(output: Any) -> Any:
    # Refactored code to handle single or tuple outputs
    return output[0] if type(output) == tuple else output


def __distribute__(params, kwargs):
    out = {}
    for key, param in params.items():
        data = kwargs[key]
        if data is inspect.Parameter.empty:
            if param.default is inspect.Parameter.empty:
                raise TypeError(f"Required parameter {key} is empty.")
            data = param.default
        out[key] = data
    return out


def __init_mp__(message_func=None, aggregate_func=None):
    # Extract parameters from message and aggregate functions if provided
    msg_params = inspect.signature(message_func).parameters if message_func else {}
    aggr_params = inspect.signature(aggregate_func).parameters if aggregate_func else {}

    msg_params = OrderedDict(msg_params)
    aggr_params = OrderedDict(aggr_params)
    if aggr_params:
        aggr_params.popitem(last=False)  # Assuming the first parameter is always 'self' or similar context

    msg_args = set(msg_params.keys()) - msg_special_args
    aggr_args = set(aggr_params.keys()) - aggr_special_args
    args = set().union(msg_args, aggr_args)

    return msg_params, aggr_params, args


def set_size(size, index, tensor, node_dim):
    if not is_tensor(tensor) and not isinstance(tensor, ndarray):
        return
    if size[index] is None:
        size[index] = tensor.shape[node_dim]
    elif size[index] != tensor.shape[node_dim]:
        raise ValueError(
            f"Encountered node tensor with size {tensor.shape[node_dim]} in dimension {node_dim}, but expected size {size[index]}.")


def __collect__(edge_index, size, kwargs, flow, node_dim, args):
    i, j = (0, 1) if flow == "target_to_source" else (1, 0)
    ij = {"_i": i, "_j": j}

    out = {}
    for arg in args:
        if arg[-2:] not in ij.keys():
            out[arg] = kwargs.get(arg, inspect.Parameter.empty)
        else:
            idx = ij[arg[-2:]]
            data = kwargs.get(arg[:-2], inspect.Parameter.empty)

            if data is inspect.Parameter.empty:
                out[arg] = data
                continue

            if isinstance(data, tuple) or isinstance(data, list):
                assert len(data) == 2
                set_size(size, 1 - idx, data[1 - idx], node_dim)
                data = data[idx]

            if not is_tensor(data) and not isinstance(data, ndarray):
                out[arg] = data
                continue

            set_size(size, idx, data, node_dim)

            if is_tensor(data):
                out[arg] = data.index_select(node_dim, edge_index[idx])
            if isinstance(data, ndarray):
                out[arg] = data.take(edge_index[idx], axis=node_dim)

    size[0] = size[1] if size[0] is None else size[0]
    size[1] = size[0] if size[1] is None else size[1]

    # Add special arguments
    out["edge_index"] = edge_index
    out["edge_index_i"] = edge_index[i]
    out["edge_index_j"] = edge_index[j]
    out["size"] = size
    out["size_i"] = size[i]
    out["size_j"] = size[j]
    out["index"] = out["edge_index_i"]
    out["dim_size"] = out["size_i"]

    return out
