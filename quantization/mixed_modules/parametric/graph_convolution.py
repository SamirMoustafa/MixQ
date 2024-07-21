from torch import tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter

from quantization.mixed_modules.base_module import MQParametricModule
from quantization.mixed_modules.parametric import MQLinear
from quantization.mixed_modules.non_parametric import MQInput, MQMessagePassing


class MQGraphConvolution(MQParametricModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 qi,
                 qo,
                 num_bits_list,
                 is_signed=False,
                 quantize_per="tensor",
                 aggr="add",
                 flow="source_to_target",
                 node_dim=0,
                 normalize=True,
                 cached=True,
                 ):
        super(MQGraphConvolution, self).__init__(qi=qi, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed)

        self.aggr = aggr
        self.lin = MQLinear(in_channels, out_channels, bias=False, qi=qi, qo=True, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
        self.e = MQInput(qi=True, num_bits_list=num_bits_list, is_signed=False, quantize_per=quantize_per)
        self.mp = MQMessagePassing(flow=flow, node_dim=node_dim, qi=False, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)

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
        self.__cache__ = [None, None]

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

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.mp.node_dim, dim_size=dim_size, reduce=self.aggr)

    def calculate_weighted_loss(self, x, edge_index, edge_weight=None):
        device = x.device
        loss = tensor([0.0], device=device)
        loss += self.e.calculate_weighted_loss(x)
        loss += self.lin.calculate_weighted_loss(x)
        x = self.lin(x)
        loss += self.mp.calculate_weighted_loss(x, edge_index, edge_weight)
        return loss / (1024 * 8)

    def select_top_k_winners(self, k):
        wining_bit_width = {}
        wining_bit_width.update({f"lin_{key}": value for key, value in self.lin.select_top_k_winners(k).items()})
        wining_bit_width.update({f"e_{key}": value for key, value in self.e.select_top_k_winners(k).items()})
        wining_bit_width.update({f"mp_{key}": value for key, value in self.mp.select_top_k_winners(k).items()})
        return wining_bit_width

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        lin_integer_operations = self.lin.estimated_bit_operation_precision(x)
        x = self.lin(x)
        mp_integer_operations = self.mp.estimated_bit_operation_precision(x, edge_index, edge_weight)
        return lin_integer_operations + mp_integer_operations
