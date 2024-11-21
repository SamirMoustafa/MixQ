from torch import tensor
from torch_scatter import scatter

from quantization.mixed_modules.base_module import MQParametricModule
from quantization.mixed_modules.parametric import MQLinear
from quantization.mixed_modules.non_parametric import MQInput, MQMessagePassing


class MQGraphSAGE(MQParametricModule):
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
                 ):
        super(MQGraphSAGE, self).__init__(qi=qi, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed)

        self.aggr = aggr
        self.mp = MQMessagePassing(flow=flow, node_dim=node_dim, qi=qi, qo=True, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
        self.lin_l = MQLinear(in_channels, out_channels, bias=False, qi=False, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
        self.lin_r = MQLinear(in_channels, out_channels, bias=False, qi=False, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)

        self.mp.message = self.message
        self.mp.aggregate = self.aggregate
        self.quantize_per = quantize_per

    def reset_parameters(self):
        super().reset_parameters()
        self.mp.reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x_l = self.mp(edge_index, x=x, edge_weight=edge_weight)
        x_l = self.lin_l(x_l)
        x = x_l + self.lin_r(x)
        return x

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.mp.node_dim, dim_size=dim_size, reduce=self.aggr)

    def calculate_weighted_loss(self, x, edge_index, edge_weight=None):
        device = x.device
        loss = tensor([0.0], device=device)
        loss += self.mp.calculate_weighted_loss(x, edge_index, edge_weight)
        x_l = self.mp(edge_index, x=x, edge_weight=edge_weight)
        loss += self.lin_l.calculate_weighted_loss(x_l)
        loss += self.lin_r.calculate_weighted_loss(x)
        return loss / (1024 * 8)

    def select_top_k_winners(self, k):
        wining_bit_width = {}
        wining_bit_width.update({f"mp_{key}": value for key, value in self.mp.select_top_k_winners(k).items()})
        wining_bit_width.update({f"lin_l_{key}": value for key, value in self.lin_l.select_top_k_winners(k).items()})
        wining_bit_width.update({f"lin_r_{key}": value for key, value in self.lin_r.select_top_k_winners(k).items()})
        return wining_bit_width

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        mp_integer_operations = self.mp.estimated_bit_operation_precision(x, edge_index, edge_weight)
        x_l = self.mp(edge_index, x=x, edge_weight=edge_weight)
        lin_l_integer_operations = self.lin_l.estimated_bit_operation_precision(x_l)
        lin_r_integer_operations = self.lin_r.estimated_bit_operation_precision(x)
        return mp_integer_operations + lin_l_integer_operations + lin_r_integer_operations
