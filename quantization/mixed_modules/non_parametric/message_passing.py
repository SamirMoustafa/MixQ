from torch import tensor

from torch_operation_counter import OperationsCounterMode

from quantization.mixed_modules.base_module import MQNonParametricModule
from quantization.message_passing_base import __collect__, __init_mp__, __process_size__, __distribute__, __single_or_tuple__
from quantization.utility import _top_k_of_softmax_of_tensor
from utility import flatten_list


class MQMessagePassing(MQNonParametricModule):
    def __init__(self, qi, qo, num_bits_list, is_signed, quantize_per, flow: str = "source_to_target", node_dim: int = 0):
        super(MQMessagePassing, self).__init__(qi=qi, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)

        self.__msg_params__, self.__aggr_params__, self.args = __init_mp__(self.message, self.aggregate)
        assert flow in ["source_to_target", "target_to_source"], f"{flow} is not a valid flow direction."
        assert node_dim >= 0, "node_dim must be non-negative."
        self.flow, self.node_dim = flow, node_dim

        self.is_signed = is_signed
        self.num_bits_list = num_bits_list
        self.quantize_per = quantize_per

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def relaxed_quantize_input(self, x, q_coeff_softmax):
        if self.qi is not None:
            xs = []
            for i, q in enumerate(self.qi):
                q.calibrate(x)
                xs += [q.fake_quantize(x) * q_coeff_softmax[i]]
            x = sum(xs)
        return x

    def relaxed_quantize_output(self, x, q_coeff_softmax):
        if self.qo is not None:
            xs = []
            for i, q in enumerate(self.qo):
                q.calibrate(x)
                xs += [q.fake_quantize(x) * q_coeff_softmax[i]]
            x = sum(xs)
        return x

    def propagate(self, edge_index, size=None, **kwargs):
        size = __process_size__(size)

        kwargs = __collect__(edge_index, size, kwargs, self.flow, self.node_dim, self.args)
        msg_kwargs = __distribute__(self.__msg_params__, kwargs)
        x = self.message(**msg_kwargs)
        x = __single_or_tuple__(x)

        aggr_kwargs = __distribute__(self.__aggr_params__, kwargs)
        x = self.aggregate(x, **aggr_kwargs)
        x = __single_or_tuple__(x)
        return x

    def message(self, x_j, edge_weight=None):
        raise NotImplementedError("Not implemented yet.")

    def aggregate(self, inputs, index, dim_size):
        raise NotImplementedError("Not implemented yet.")

    def forward(self, edge_index, size=None, **kwargs):
        if self.qi is not None and "x" in kwargs:
            x = kwargs["x"]
            qi_coeff_softmax = self.qi_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_input(x, qi_coeff_softmax)
            kwargs["x"] = x

        x = self.propagate(edge_index, size, **kwargs)

        if self.qo is not None:
            qo_coeff_softmax = self.qo_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_output(x, qo_coeff_softmax)
        return x

    def calculate_weighted_loss(self, x, edge_index, edge_weight=None):
        device = x.device
        loss = tensor([0.0], device=device)
        bits_tensor = tensor([self.num_bits_list], device=device, dtype=x.dtype)

        if self.qi is not None:
            qi_coeff_softmax = self.qi_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_input(x, qi_coeff_softmax)
            loss += (bits_tensor * qi_coeff_softmax).sum() * x.numel()

        x = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.qo is not None:
            qo_coeff_softmax = self.qo_relaxed_coeff.softmax(dim=0)
            x = self.relaxed_quantize_output(x, qo_coeff_softmax)
            loss += (bits_tensor * qo_coeff_softmax).sum() * x.numel()

        return loss / (1024 * 8)

    def select_top_k_winners(self, k):
        wining_bit_width = {"qi": [],  "qo": []}
        if self.qi is not None:
            top_k_qi = _top_k_of_softmax_of_tensor(self.qi_relaxed_coeff, k)
            wining_bit_width["qi"] = [self.num_bits_list[i] for i in top_k_qi]
        else:
            wining_bit_width["qi"] = [None for _ in range(k)]

        if self.qo is not None:
            top_k_qo = _top_k_of_softmax_of_tensor(self.qo_relaxed_coeff, k)
            wining_bit_width["qo"] = [self.num_bits_list[i] for i in top_k_qo]
        else:
            wining_bit_width["qo"] = [None for _ in range(k)]
        return wining_bit_width

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        wining_bit_width = [*filter(None, flatten_list([*self.select_top_k_winners(1).values()]))]
        if len(wining_bit_width) > 0:
            expected_bit_width = sum(wining_bit_width) / len(wining_bit_width)
            with OperationsCounterMode(self) as ops_counter:
                self.propagate(edge_index, x=x, edge_weight=edge_weight)
            return ops_counter.total_main_operation * expected_bit_width
        else:
            return 0
