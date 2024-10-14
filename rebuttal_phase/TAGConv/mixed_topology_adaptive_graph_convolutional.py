from torch import tensor
from torch.nn import ModuleList
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter

from quantization.mixed_modules.base_module import MQParametricModule
from quantization.mixed_modules.parametric import MQLinear
from quantization.mixed_modules.non_parametric import MQInput, MQMessagePassing


class MQTopologyAdaptiveGraphConvolution(MQParametricModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 k,
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
        super(MQTopologyAdaptiveGraphConvolution, self).__init__(qi=qi, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed)

        self.aggr = aggr
        self.k = k
        self.lins = ModuleList([MQLinear(in_channels, out_channels, bias=False, qi=qi, qo=True, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
                                for _ in range(k + 1)])
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
        for lin in self.lins:
            lin.reset_parameters()
        self.mp.reset_parameters()
        self.__cache__ = [None, None]

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
        edge_weight = self.e(edge_weight) if edge_weight is not None else None
        x_out = self.lins[0](x)
        for lin in self.lins[1:]:
            x = self.mp(edge_index, x=x, edge_weight=edge_weight)
            x_out = x_out + lin.forward(x)
        return x_out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.mp.node_dim, dim_size=dim_size, reduce=self.aggr)

    def calculate_weighted_loss(self, x, edge_index, edge_weight=None):
        device = x.device
        loss = tensor([0.0], device=device)
        loss += self.e.calculate_weighted_loss(edge_index) if edge_weight is not None else 0
        loss += self.lins[0].calculate_weighted_loss(x)
        x_out = self.lins[0](x)
        for lin in self.lins[1:]:
            loss += self.mp.calculate_weighted_loss(x, edge_index, edge_weight)
            x = self.mp(edge_index, x=x, edge_weight=edge_weight)
            loss += lin.calculate_weighted_loss(x)
            x_out = x_out + lin.forward(x)
            loss += lin.calculate_weighted_loss(x)
        return loss / (1024 * 8)

    def select_top_k_winners(self, k):
        wining_bit_width = {}
        for i, lin in enumerate(self.lins):
            wining_bit_width.update({f"lin_{i}_{key}": value for key, value in lin.select_top_k_winners(k).items()})
        wining_bit_width.update({f"e_{key}": value for key, value in self.e.select_top_k_winners(k).items()})
        wining_bit_width.update({f"mp_{key}": value for key, value in self.mp.select_top_k_winners(k).items()})
        return wining_bit_width

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        integer_operations = self.lins[0].estimated_bit_operation_precision(x)
        x_out = self.lins[0](x)
        for lin in self.lins[1:]:
            integer_operations += self.mp.estimated_bit_operation_precision(x, edge_index, edge_weight)
            x = self.mp(edge_index, x=x, edge_weight=edge_weight)
            integer_operations += lin.estimated_bit_operation_precision(x)
            x_out = x_out + lin.forward(x)
        return integer_operations
