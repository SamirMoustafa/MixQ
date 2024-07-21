from io import BytesIO

import pygraphviz
from PIL import Image
from matplotlib import pyplot as plt

from torch import cuda, randn, randint, int64
from torch.nn import Module, ModuleList
from torch.nn.functional import dropout
from torch.utils._pytree import tree_map

# ! require torch_traverser package to be installed
from torch_traverser.traverser import traverser_module
from torch_traverser.recorder_tensor import cast_to_recoder_tensor
# ! require torch_operation_counter package to be installed
from torch_operation_counter import OperationsCounterMode

from quantization.fixed_modules.non_parametric import QReLU
from quantization.fixed_modules.parametric import QLinearBatchNormReLU, QGraphIsomorphism, QLinear


class OperationsContextManager(OperationsCounterMode):
    def __init__(self):
        super(OperationsContextManager, self).__init__()
        self.number_of_operations = None

    def get_parameters(self):
        return {"number_of_operations": self.number_of_operations}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = super(OperationsContextManager, self).__torch_dispatch__(func, types, args, kwargs)
        self.number_of_operations = self.total_operations
        return tree_map(lambda x: cast_to_recoder_tensor(x), out)


class QGIN(Module):
    def __init__(self,
                 in_channels,
                 num_layers,
                 hidden_channels,
                 out_channels,
                 bit_widths,
                 dropout_p=0.2,
                 ):
        super(QGIN, self).__init__()

        self.dropout_p = dropout_p

        self.convs = ModuleList()
        for i in range(num_layers):
            is_first_layer = i == 0
            mlp_i = QLinearBatchNormReLU(hidden_channels if i > 0 else in_channels,
                                         hidden_channels,
                                         qi=False,
                                         qo=True,
                                         num_bits=bit_widths[i][0],
                                         )
            gin_i = QGraphIsomorphism(nn=mlp_i, qi=is_first_layer, qo=True, num_bits=bit_widths[i][1])
            self.convs.append(gin_i)

        self.lin1 = QLinear(in_features=hidden_channels,
                            out_features=hidden_channels,
                            qi=True,
                            qo=True,
                            num_bits=bit_widths[num_layers],
                            )
        self.relu = QReLU(num_bits=bit_widths[num_layers + 1])
        self.lin2 = QLinear(in_features=hidden_channels,
                            out_features=out_channels,
                            qi=False,
                            qo=True,
                            num_bits=bit_widths[num_layers + 2],
                            )
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def set_forward_func(self, forward_func: callable):
        self.forward = forward_func

    def full_precision_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

    def simulated_quantize_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv.simulated_quantize_forward(x, edge_index)
            x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin1.simulated_quantize_forward(x)
        x = self.relu.simulated_quantize_forward(x)
        x = self.lin2.simulated_quantize_forward(x)
        return x

    def freeze(self):
        qo = self.convs[0].qo
        for i, conv in enumerate(self.convs):
            conv.freeze(qi=qo) if i > 0 else conv.freeze()
            qo = conv.qo
        self.lin1.freeze(qi=qo)
        self.relu.freeze(qi=self.lin1.qo)
        self.lin2.freeze(qi=self.lin1.qo)

    def quantize_inference(self, x, edge_index):
        qx = self.convs[0].qi.quantize(x)
        for conv in self.convs:
            qx = conv.quantize_inference(qx, edge_index)
        qx = self.lin1.quantize_inference(qx)
        qx = self.relu.quantize_inference(qx)
        qx = self.lin2.quantize_inference(qx)
        x = self.lin2.qo.dequantize(qx)
        return x


if __name__ == '__main__':
    device_ = "cpu" if cuda.is_available() else "cpu"

    # Reddit dataset settings
    n = 232_965
    e = 114_615_892 + n
    num_feature = 602
    out_feature = 41

    # Define the model
    num_layers = 3
    hidden_channels = 128
    bit_widths = [[[-1, -1, -1],  # MLP Bit-widths
                   [-1, -1]  # Graph Isomorphism Bit-widths
                   ] for _ in range(num_layers)]
    bit_widths += [[-1, -1, -1],
                   [-1],
                   [-1, -1, -1],
                   ]
    model = QGIN(in_channels=num_feature, hidden_channels=hidden_channels, out_channels=out_feature,
                 num_layers=num_layers, bit_widths=bit_widths)

    # Prepare model input
    model_input = {"x": randn(n, num_feature, device=device_),
                   "edge_index": randint(1, n, size=(2, e), dtype=int64, device=device_),
                   }
    model.set_forward_func(model.full_precision_forward)

    model_graph = traverser_module(model=model,
                                   input_data=model_input,
                                   context_manager_for_forward=OperationsContextManager,
                                   device_=device_,
                                   depth=2,
                                   )

    # Extract nodes attributes for labels
    names = {node: attrs["name"] for node, attrs in model_graph.nodes_attributes.items() if "name" in attrs}
    input_shapes = {node: attrs["input"] for node, attrs in model_graph.nodes_attributes.items() if "input" in attrs}
    output_shapes = {node: attrs["output"] for node, attrs in model_graph.nodes_attributes.items() if "output" in attrs}
    operations = {k: f"{v['number_of_operations']}" for k, v in model_graph.nodes_attributes.items() if
                  "number_of_operations" in v}

    edges = model_graph.computational_dag.to_edge_list()
    G = pygraphviz.AGraph(directed=True, rankdir="BT")
    [G.add_edge(edge[0], edge[1]) for edge in edges]

    node_styles = {
        "style": "filled, rounded",
        "fillcolor": "#9aceff",
        "fontcolor": "black",
        "fontsize": "18",
        "shape": "box",
    }

    for node in G.nodes():
        node_int = int(node)
        G.get_node(node).attr.update(node_styles)
        node_label = f"<B>{names.get(node_int, str(node_int)).replace('Q', '')}</B>"

        details = []
        if node_int in input_shapes:
            input_shapes_str = '; '.join(map(str, input_shapes[node_int]))
            details.append(f"Input: {input_shapes_str}")
        if node_int in output_shapes:
            output_shapes_str = '; '.join(map(str, output_shapes[node_int]))
            details.append(f"Output: {output_shapes_str}")
        if node_int in operations:
            details.append(f"Operations: {operations[node_int]}")

        if details:
            node_label += "<BR/>" + "<BR/>".join(details)

        G.get_node(node).attr["label"] = f"<<FONT POINT-SIZE='12'>{node_label}</FONT>>"
        G.get_node(node).attr["shape"] = "box" if node_int in operations else "rect"

    G.layout(prog="dot")
    png_buffer = BytesIO()
    G.draw(png_buffer, format="png", prog="dot")
    G.draw(f"number_of_ops_{num_layers}_layer_gin_depth_2.pdf", format="pdf", prog="dot")
    png_buffer.seek(0)
    graph_image = Image.open(png_buffer)
    image_width, image_height = graph_image.size
    desired_dpi = 100
    fig_width = image_width / desired_dpi
    fig_height = image_height / desired_dpi
    plt.figure(figsize=(fig_width, fig_height), dpi=desired_dpi)
    plt.imshow(graph_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
