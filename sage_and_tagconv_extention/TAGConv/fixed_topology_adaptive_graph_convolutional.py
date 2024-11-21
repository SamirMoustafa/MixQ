from torch.nn import ModuleList
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter

from quantization.fixed_modules.base_module import QParametricModule
from quantization.fixed_modules.parametric import QLinear
from quantization.fixed_modules.non_parametric import QInput, QMessagePassing


class QTopologyAdaptiveGraphConvolution(QParametricModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 k,
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
        super(QTopologyAdaptiveGraphConvolution, self).__init__(qi=qi, qo=qo, num_bits=[-1, -1], is_signed=is_signed)

        self.k = k
        self.aggr = aggr
        self.lins = ModuleList([
            QLinear(in_channels, out_channels, bias=False, qi=qi, qo=True, num_bits=num_bits[i:i+3], is_signed=is_signed, quantize_per=quantize_per)
            for i in range(0, 2 * k + 1, 3)])
        self.e = QInput(qi=True, num_bits=[num_bits[3 * (k + 1)], ], is_signed=False, quantize_per=quantize_per)
        self.mp = QMessagePassing(flow=flow, node_dim=node_dim, qi=False, qo=False, num_bits=num_bits[3 * (k + 1) + 1: 3 * (k + 1) + 3], is_signed=is_signed, quantize_per=quantize_per)
        # TODO: There should be quantization for the output of the message passing layer
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
        for lin in self.lins:
            lin.reset_parameters()
        self.mp.reset_parameters()
        self.e.reset_parameters()
        self.__cache__ = [None, None]

    def freeze(self, qi=None, qo=None):
        raise NotImplementedError("Freezing the quantization of the GraphSAGE layer is not supported.")

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize and not self.cache:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
        if self.cache:
            if self.__cache__[1] is None:
                if self.normalize:
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
                self.__cache__ = [edge_index, edge_weight]
            else:
                edge_index, edge_weight = self.__cache__

        x_out = self.lins[0](x)
        for lin in self.lins[1:]:
            x = self.mp(edge_index, x=x, edge_weight=edge_weight)
            x_out = x_out + lin.forward(x)
        return x_out

    def simulated_quantize_forward(self, x, edge_index, edge_weight=None):
        if self.normalize and not self.cache:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
        if self.cache:
            if self.__cache__[1] is None:
                if self.normalize:
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)
                self.__cache__ = [edge_index, edge_weight]
            else:
                edge_index, edge_weight = self.__cache__
        edge_weight = self.e.simulated_quantize_forward(edge_weight) if edge_weight is not None else None

        x_out = self.lins[0].simulated_quantize_forward(x)
        for lin in self.lins[1:]:
            x = self.mp.simulated_quantize_forward(edge_index, x=x, edge_weight=edge_weight)
            x_out = x_out + lin.simulated_quantize_forward(x)
        return x_out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.mp.node_dim, dim_size=dim_size, reduce=self.aggr)

    def quantize_inference(self, x, edge_index, edge_weight=None, size=None):
        raise NotImplementedError("Quantizing the inference of the GraphSAGE layer is not supported.")

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        if self.cache:
            if self.__cache__[1] is None:
                self.__cache__ = [edge_index, edge_weight]
            else:
                edge_index, edge_weight = self.__cache__
        elif self.normalize:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, improved=True, flow=self.mp.flow)

        integer_operations = self.lins[0].estimated_bit_operation_precision(x)
        x_out = self.lins[0](x)
        for lin in self.lins[1:]:
            integer_operations += self.mp.estimated_bit_operation_precision(x, edge_index, edge_weight)
            x = self.mp(edge_index, x=x, edge_weight=edge_weight)
            integer_operations += lin.estimated_bit_operation_precision(x)
            x_out = x_out + lin(x)
        return integer_operations
