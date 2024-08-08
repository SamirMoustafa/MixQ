from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter

from quantization.fixed_modules.base_module import QParametricModule
from quantization.fixed_modules.parametric import QLinear
from quantization.fixed_modules.non_parametric import QInput, QMessagePassing
from quantization.fixed_modules.non_parametric.message_passing import MaskQuantMessagePassing


class QGraphConvolution(QParametricModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 qi,
                 qo,
                 num_bits,
                 is_signed=False,
                 quantize_per="tensor",
                 aggr="add",
                 flow="source_to_target",
                 node_dim=0,
                 normalize=True,
                 cached=True,
                 ):
        super(QGraphConvolution, self).__init__(qi=qi, qo=qo, num_bits=[-1, -1], is_signed=is_signed)

        self.aggr = aggr
        self.lin = QLinear(in_channels, out_channels, bias=False, qi=qi, qo=True, num_bits=num_bits[:3], is_signed=is_signed, quantize_per=quantize_per)
        self.e = QInput(qi=True, num_bits=[num_bits[3], ], is_signed=False, quantize_per=quantize_per)
        self.mp = QMessagePassing(flow=flow, node_dim=node_dim, qi=False, qo=qo, num_bits=num_bits[4:6], is_signed=is_signed, quantize_per=quantize_per)

        self.mp.message = self.message
        self.mp.aggregate = self.aggregate
        self.quantize_per = quantize_per

        self.normalize = normalize
        self.cache = cached
        if cached:
            self.__cache__ = [None, None]  # (node with self-loops, node weights)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.mp.reset_parameters()
        self.e.reset_parameters()
        self.__cache__ = [None, None]

    def freeze(self, qi=None, qo=None):
        if self.quantize_per != 'tensor':
            raise NotImplementedError("Only tensor-wise quantization is supported for now.")
        if qi is not None:
            self.qi = qi.copy()
        elif self.lin.qi is not None:
            self.qi = self.lin.qi.copy()

        if self.mp.qo is not None:
            self.qo = self.mp.qo.copy()

        self.lin.freeze(qi=self.qi)
        self.mp.freeze(qi=self.lin.qo, qe=self.e.qi)
        self.e.freeze()

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize and not self.cache:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
        if self.cache:
            if self.__cache__[0] is None:
                if self.normalize:
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
                self.__cache__ = [edge_index, edge_weight]
            else:
                edge_index, edge_weight = self.__cache__

        x = self.lin(x)
        x = self.mp(edge_index, x=x, edge_weight=edge_weight)
        return x

    def simulated_quantize_forward(self, x, edge_index, edge_weight=None):
        if self.normalize and not self.cache:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
        if self.cache:
            if self.__cache__[0] is None:
                if self.normalize:
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
                self.__cache__ = [edge_index, edge_weight]
            else:
                edge_index, edge_weight = self.__cache__
        edge_weight = self.e.simulated_quantize_forward(edge_weight)

        x = self.lin.simulated_quantize_forward(x)
        x = self.mp.simulated_quantize_forward(edge_index, x=x, edge_weight=edge_weight)
        return x

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.mp.node_dim, dim_size=dim_size, reduce=self.aggr)

    def quantize_inference(self, x, edge_index, edge_weight=None, size=None):
        if self.cache:
            if self.__cache__[0] is None:
                self.__cache__ = [edge_index, edge_weight]
            else:
                edge_index, edge_weight = self.__cache__
        elif self.normalize:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)

        edge_weight = self.e.quantize_inference(edge_weight)
        x = self.lin.quantize_inference(x)
        x = self.mp.quantize_inference(edge_index, x=x, edge_weight=edge_weight)
        return x

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        if self.cache:
            if self.__cache__[0] is None:
                self.__cache__ = [edge_index, edge_weight]
            else:
                edge_index, edge_weight = self.__cache__
        elif self.normalize:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)

        lin_integer_operations = self.lin.estimated_bit_operation_precision(x)
        x = self.lin(x)
        mp_integer_operations = self.mp.estimated_bit_operation_precision(x, edge_index, edge_weight)
        return lin_integer_operations + mp_integer_operations


class MaskQuantGraphConvolution(QGraphConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
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
                 normalize=True,
                 cached=True,
                 ):
        super(MaskQuantGraphConvolution, self).__init__(in_channels=in_channels, out_channels=out_channels, qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, quantize_per=quantize_per, aggr=aggr, flow=flow, node_dim=node_dim, normalize=normalize, cached=cached)
        self.mp = MaskQuantMessagePassing(qi=qi, qo=qo, num_bits=num_bits, is_signed=is_signed, use_momentum=quant_use_momentum, momentum=quant_momentum, percentile=quant_percentile, sample_ratio=quant_sample_ratio, flow=flow, node_dim=node_dim)
        self.mp.message = self.message
        self.mp.aggregate = self.aggregate

    def simulated_quantize_forward(self, x, edge_index, binary_mask, edge_weight=None):
        if self.normalize and not self.cache:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
        if self.cache:
            if self.__cache__[0] is None:
                if self.normalize:
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
                self.__cache__ = [edge_index, edge_weight]
            else:
                edge_index, edge_weight = self.__cache__
        edge_weight = self.e.simulated_quantize_forward(edge_weight)

        x = self.lin.simulated_quantize_forward(x)
        x = self.mp.simulated_quantize_forward(edge_index, x=x, edge_weight=edge_weight, binary_mask=binary_mask)
        return x
