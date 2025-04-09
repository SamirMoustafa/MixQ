# This implementation of GraphSAGE is limited to `add` only as aggregation and without the bias.
# It's a proof of concept to show how to implement a fixed-point GraphSAGE layer.
from torch_scatter import scatter

from quantization.fixed_modules.base_module import QParametricModule
from quantization.fixed_modules.parametric import QLinear
from quantization.fixed_modules.non_parametric import QInput, QMessagePassing


class QGraphSAGE(QParametricModule):
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
                 ):
        super(QGraphSAGE, self).__init__(qi=qi, qo=qo, num_bits=[-1, -1], is_signed=is_signed)

        self.aggr = aggr
        self.mp = QMessagePassing(flow=flow, node_dim=node_dim, qi=qi, qo=True, num_bits=num_bits[:2], is_signed=is_signed, quantize_per=quantize_per)
        self.lin_l = QLinear(in_channels, out_channels, bias=False, qi=False, qo=qo, num_bits=num_bits[2:5], is_signed=is_signed, quantize_per=quantize_per)
        self.lin_r = QLinear(in_channels, out_channels, bias=False, qi=False, qo=qo, num_bits=num_bits[5:8], is_signed=is_signed, quantize_per=quantize_per)

        self.mp.message = self.message
        self.mp.aggregate = self.aggregate
        self.quantize_per = quantize_per

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.mp.reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def freeze(self, qi=None, qo=None):
        raise NotImplementedError("Freezing the quantization of the GraphSAGE layer is not supported.")

    def forward(self, x, edge_index):
        x_l = self.mp(edge_index, x=x)
        x_l = self.lin_l(x_l)
        x = x_l + self.lin_r(x)
        return x

    def simulated_quantize_forward(self, x, edge_index, edge_weight=None):
        x_l = self.mp.simulated_quantize_forward(edge_index, x=x, edge_weight=edge_weight)
        x_l = self.lin_l.simulated_quantize_forward(x_l)
        x = x_l + self.lin_r.simulated_quantize_forward(x)
        return x

    def message(self, x_j, edge_weight=None):
        return x_j

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.mp.node_dim, dim_size=dim_size, reduce=self.aggr)

    def quantize_inference(self, x, edge_index, edge_weight=None, size=None):
        raise NotImplementedError("Quantizing the inference of the GraphSAGE layer is not supported.")

    def estimated_bit_operation_precision(self, x, edge_index, edge_weight=None):
        mp_integer_operations = self.mp.estimated_bit_operation_precision(x, edge_index, edge_weight)
        x_l = self.mp(edge_index, x=x, edge_weight=edge_weight)
        lin_l_integer_operations = self.lin_l.estimated_bit_operation_precision(x_l)
        lin_r_integer_operations = self.lin_r.estimated_bit_operation_precision(x)
        return mp_integer_operations + lin_l_integer_operations + lin_r_integer_operations
