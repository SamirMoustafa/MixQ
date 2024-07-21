from torch_scatter import scatter

from quantization.fixed_modules.base_module import QParametricModule
from quantization.fixed_modules.non_parametric.message_passing import QMessagePassing, MaskQuantMessagePassing


class QGraphIsomorphism(QParametricModule):
    def __init__(self,
                 nn: callable,
                 qi,
                 qo,
                 num_bits,
                 is_signed=False,
                 quantize_per="tensor",
                 aggr="add",
                 flow="source_to_target",
                 node_dim=0,
                 ):
        super(QGraphIsomorphism, self).__init__(qi=False, qo=False, num_bits=num_bits, is_signed=is_signed, quantize_per=quantize_per)
        self.aggr = aggr
        self.mp = QMessagePassing(flow=flow, node_dim=node_dim, qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, quantize_per=quantize_per)
        self.nn = nn

        self.mp.message = self.message
        self.mp.aggregate = self.aggregate
        self.quantize_per = quantize_per

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.mp.reset_parameters()
        self.nn.reset_parameters()

    def freeze(self, qi=None, qo=None):
        if self.quantize_per != 'tensor':
            raise NotImplementedError("Only tensor-wise quantization is supported for now.")
        self.nn.freeze(qi=self.mp.qo)

        if qi is not None:
            self.qi = qi.copy()
        elif self.mp.qi is not None:
            self.qi = self.mp.qi.copy()

        if self.nn.qo is not None:
            self.qo = self.nn.qo.copy()

        self.mp.freeze(qi=self.qi)

    def forward(self, x, edge_index):
        x = self.mp(edge_index, x=x)
        x = self.nn(x)
        return x

    def simulated_quantize_forward(self, x, edge_index):
        x = self.mp.simulated_quantize_forward(edge_index, x=x)
        x = self.nn.simulated_quantize_forward(x)
        return x

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.mp.node_dim, dim_size=dim_size, reduce=self.aggr)

    def quantize_inference(self, x, edge_index, size=None):
        x = self.mp.quantize_inference(edge_index, x=x)
        x = self.nn.quantize_inference(x)
        return x

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        mp_integer_operations = self.mp.estimated_bit_operation_precision(x, edge_index, edge_weight)
        x = self.mp(edge_index, x=x)
        nn_integer_operations = self.nn.estimated_bit_operation_precision(x)
        return mp_integer_operations + nn_integer_operations


class MaskQuantGraphIsomorphism(QGraphIsomorphism):
    def __init__(self,
                 nn: callable,
                 qi,
                 qo,
                 num_bits,
                 is_signed=False,
                 quantize_per="tensor",
                 quant_use_momentum: bool = True,
                 quant_momentum: float = 0.1,
                 quant_percentile: float = 0.99,
                 quant_sample_ratio: float = 1.0,
                 aggr="add",
                 flow="source_to_target",
                 node_dim=0,
                 ):
        super(MaskQuantGraphIsomorphism, self).__init__(nn=nn, qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, aggr=aggr, flow=flow, node_dim=node_dim, quantize_per=quantize_per)
        self.mp = MaskQuantMessagePassing(qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, use_momentum=quant_use_momentum, momentum=quant_momentum, percentile=quant_percentile, sample_ratio=quant_sample_ratio, flow=flow, node_dim=node_dim)
        self.mp.message = self.message
        self.mp.aggregate = self.aggregate

    def simulated_quantize_forward(self, x, edge_index, binary_mask):
        x = self.mp.simulated_quantize_forward(edge_index, x=x, binary_mask=binary_mask)
        x = self.nn.simulated_quantize_forward(x)
        return x

    def quantize_inference(self, x, edge_index, size=None):
        x = self.mp.quantize_inference(edge_index, x=x)
        x = self.nn.quantize_inference(x)
        return x


class MaskQuantGraphIsomorphismWithSkipConnections(MaskQuantGraphIsomorphism):
    # Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, 2020
    def __init__(self,
                 nn: callable,
                 qi,
                 qo,
                 num_bits,
                 is_signed=False,
                 quantize_per="tensor",
                 quant_use_momentum: bool = True,
                 quant_momentum: float = 0.1,
                 quant_percentile: float = 0.99,
                 quant_sample_ratio: float = 1.0,
                 aggr="add",
                 flow="source_to_target",
                 node_dim=0,
                 ):
        super(MaskQuantGraphIsomorphismWithSkipConnections, self).__init__(nn=nn, qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, quantize_per=quantize_per, quant_use_momentum=quant_use_momentum, quant_momentum=quant_momentum, quant_percentile=quant_percentile, quant_sample_ratio=quant_sample_ratio, aggr=aggr, flow=flow, node_dim=node_dim)

    def forward(self, x, edge_index):
        propagated_x = self.mp(edge_index, x=x)
        x = self.nn(x + propagated_x)
        return x

    def simulated_quantize_forward(self, x, edge_index, binary_mask):
        propagated_x = self.mp.simulated_quantize_forward(edge_index, x=x, binary_mask=binary_mask)
        x = self.nn.simulated_quantize_forward(x + propagated_x)
        return x

    def freeze(self, qi=None, qo=None):
        raise NotImplementedError("Not implemented, Skip connections are not supported for quantize inference yet.")

    def quantize_inference(self, x, edge_index, size=None):
        raise NotImplementedError("Not implemented, Skip connections are not supported for quantize inference yet.")
