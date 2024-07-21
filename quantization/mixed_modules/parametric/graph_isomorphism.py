from torch import tensor
from torch_scatter import scatter

from quantization.mixed_modules.base_module import MQParametricModule
from quantization.mixed_modules.non_parametric.message_passing import MQMessagePassing


class MQGraphIsomorphism(MQParametricModule):
    def __init__(self,
                 nn: callable,
                 qi,
                 qo,
                 num_bits_list,
                 is_signed=False,
                 quantize_per="tensor",
                 aggr="add",
                 flow="source_to_target",
                 node_dim=0,
                 ):
        super(MQGraphIsomorphism, self).__init__(qi=False, qo=False, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
        self.aggr = aggr
        self.mp = MQMessagePassing(flow=flow, node_dim=node_dim, qi=qi, qo=qo, num_bits_list=num_bits_list, is_signed=is_signed, quantize_per=quantize_per)
        self.nn = nn

        self.mp.message = self.message
        self.mp.aggregate = self.aggregate
        self.quantize_per = quantize_per

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.mp.reset_parameters()
        self.nn.reset_parameters()

    def forward(self, x, edge_index):
        x = self.mp(edge_index, x=x)
        x = self.nn(x)
        return x

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.mp.node_dim, dim_size=dim_size, reduce=self.aggr)

    def calculate_weighted_loss(self, x, edge_index, edge_weight=None):
        device = x.device
        loss = tensor([0.0], device=device)
        loss += self.mp.calculate_weighted_loss(x, edge_index, edge_weight)
        x = self.mp(edge_index, x=x)
        loss += self.nn.calculate_weighted_loss(x)
        return loss / (1024 * 8)

    def select_top_k_winners(self, k):
        wining_bit_width = {}
        wining_bit_width.update({f"nn_{key}": value for key, value in self.nn.select_top_k_winners(k).items()})
        wining_bit_width.update({f"mp_{key}": value for key, value in self.mp.select_top_k_winners(k).items()})
        return wining_bit_width

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        mp_integer_operations = self.mp.estimated_bit_operation_precision(x, edge_index, edge_weight)
        x = self.mp(edge_index, x=x)
        nn_integer_operations = self.nn.estimated_bit_operation_precision(x)
        return mp_integer_operations + nn_integer_operations

