from torch import tensor, index_select
from torch_operation_counter import OperationsCounterMode

from quantization.fixed_modules.base_module import QNonParametricModule, QMinMaxRangesNonParametricModule
from quantization.functional import define_quantization_ranges
from quantization.message_passing_base import __collect__, __init_mp__, __process_size__, __distribute__, __single_or_tuple__


class QMessagePassing(QNonParametricModule):
    def __init__(self, qi, qo, num_bits, is_signed: bool = False, quantize_per: str = "tensor", flow: str = "source_to_target", node_dim: int = 0):
        super(QMessagePassing, self).__init__(qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, quantize_per=quantize_per)

        self.__msg_params__, self.__aggr_params__, self.args = __init_mp__(self.message, self.aggregate)
        assert flow in ["source_to_target", "target_to_source"], f"{flow} is not a valid flow direction."
        assert node_dim >= 0, "node_dim must be non-negative."
        self.flow, self.node_dim = flow, node_dim

        self.register_buffer("M", tensor([], requires_grad=False))
        self.is_signed = is_signed
        self.num_bits = num_bits
        self.qe = None
        self.quantize_per = quantize_per

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.M.data = tensor([], requires_grad=False)
        self.qe = None

    def freeze(self, qi=None, qo=None, qe=None):
        if self.quantize_per != 'tensor':
            raise NotImplementedError("Only tensor-wise quantization is supported for now.")
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        if qe is not None:
            self.qe = qe

        qi_scale = self.qi.scale if self.qi is not None else 1.0
        qo_scale = self.qo.scale if self.qo is not None else 1.0
        qe_scale = self.qe.scale if self.qe is not None else 1.0
        self.M.data = (qi_scale * qe_scale / qo_scale).data

    def forward(self, edge_index, size=None, **kwargs):
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

    def simulated_quantize_forward(self, edge_index, size=None, **kwargs):
        if self.qi is not None and "x" in kwargs:
            x = kwargs["x"]
            self.qi.calibrate(x)
            x = self.qi.fake_quantize(x)
            kwargs["x"] = x

        x = self.forward(edge_index, size, **kwargs)

        if self.qo is not None:
            self.qo.calibrate(x)
            x = self.qo.fake_quantize(x)
        return x

    def quantize_inference(self, edge_index, size=None, **kwargs):
        qmin, qmax = define_quantization_ranges(self.num_bits[1], signed=self.is_signed)
        if "x" in kwargs and self.qi is not None:
            x = kwargs["x"]
            x = x - self.qi.zero_point
            if self.qe is not None:
                x = x - self.qe.zero_point
            kwargs["x"] = x
        x = self.forward(edge_index, size, **kwargs)
        x = self.M * x
        x = x.round()
        x = x + self.qo.zero_point
        x = x.clamp(qmin, qmax)
        return x

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        expected_bit_width = []
        if self.qi is not None:
            expected_bit_width += [self.qi.num_bits]
        if self.qo is not None:
            expected_bit_width += [self.qo.num_bits]
        if len(expected_bit_width) > 0:
            expected_bit_width = sum(expected_bit_width) / len(expected_bit_width)
            with OperationsCounterMode(self) as ops_counter:
                self.forward(edge_index, x=x, edge_weight=edge_weight)
            return ops_counter.total_main_operation * expected_bit_width
        else:
            return 0

class MaskQuantMessagePassing(QMinMaxRangesNonParametricModule):
    # Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, 2020
    def __init__(self, qi, qo, num_bits, is_signed, use_momentum=True, momentum: float = 0.1, percentile: float = 0.99, sample_ratio: float = 1.0, flow: str = "source_to_target", node_dim: int = 0):
        super(MaskQuantMessagePassing, self).__init__(qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, use_momentum=use_momentum, momentum=momentum, percentile=percentile, sample_ratio=sample_ratio)
        self.__msg_params__, self.__aggr_params__, self.args = __init_mp__(self.message, self.aggregate)
        assert flow in ["source_to_target", "target_to_source"], f"{flow} is not a valid flow direction."
        assert node_dim >= 0, "node_dim must be non-negative."
        self.flow, self.node_dim = flow, node_dim

        self.register_buffer("M", tensor([], requires_grad=False))
        self.is_signed = is_signed
        self.num_bits = num_bits
        self.qe = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.M.data = tensor([], requires_grad=False)
        self.qe = None

    def freeze(self, qi=None, qo=None, qe=None):
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        if qe is not None:
            self.qe = qe

        qi_scale = self.qi.scale if self.qi is not None else 1.0
        qo_scale = self.qo.scale if self.qo is not None else 1.0
        qe_scale = self.qe.scale if self.qe is not None else 1.0
        self.M.data = (qi_scale * qe_scale / qo_scale).data

    def forward(self, edge_index, size=None, **kwargs):
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

    def simulated_quantize_forward(self, edge_index, binary_mask, size=None, **kwargs):
        size = __process_size__(size)
        kwargs = __collect__(edge_index, size, kwargs, self.flow, self.node_dim, self.args)
        msg_kwargs = __distribute__(self.__msg_params__, kwargs)
        x = self.message(**msg_kwargs)
        x = __single_or_tuple__(x)

        if self.qi is not None:
            if self.training:
                edge_mask = index_select(binary_mask, 0, edge_index[0])
                self.qi.calibrate(x[~edge_mask])
                x[~edge_mask] = self.qi.fake_quantize(x[~edge_mask])
            else:
                x = self.qi.fake_quantize(x)

        aggr_kwargs = __distribute__(self.__aggr_params__, kwargs)
        x = self.aggregate(x, **aggr_kwargs)
        x = __single_or_tuple__(x)

        if self.qo is not None:
            if self.training:
                self.qo.calibrate(x[~binary_mask])
                x[~binary_mask] = self.qo.fake_quantize(x[~binary_mask])
            else:
                x = self.qo.fake_quantize(x)

        return x

    def quantize_inference(self, edge_index, size=None, **kwargs):
        qmin, qmax = define_quantization_ranges(self.num_bits[1], signed=self.is_signed)
        if "x" in kwargs and self.qi is not None:
            x = kwargs["x"]
            x = x - self.qi.zero_point
            kwargs["x"] = x
        x = self.forward(edge_index, size, **kwargs)
        x = self.M * x
        x = x.round()
        x = x + self.qo.zero_point
        x = x.clamp(qmin, qmax)
        return x