from io import BytesIO
from typing import List

import pygraphviz
from PIL import Image
from matplotlib import pyplot as plt

from torch import cuda, randn, randint, int64, ones
from torch.nn import Module
from torch.nn.functional import dropout
from torch.utils._pytree import tree_map

# ! require torch_traverser package to be installed
from torch_traverser.traverser import traverser_module
from torch_traverser.recorder_tensor import cast_to_recoder_tensor
# ! require torch_operation_counter package to be installed
from torch_operation_counter import OperationsCounterMode

from quantization.fixed_modules.non_parametric import QReLU
from quantization.fixed_modules.parametric import QGraphConvolution


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


class QGCN(Module):
    def __init__(self, num_channels: int, out_channels: int, bit_widths: List[List[int]]):
        super(QGCN, self).__init__()
        self.gcn_1 = QGraphConvolution(in_channels=num_channels,
                                       out_channels=128,
                                       normalize=False,
                                       qi=True,
                                       qo=True,
                                       num_bits=bit_widths[0],
                                       )
        self.relu_1 = QReLU(num_bits=bit_widths[1])
        self.gcn_2 = QGraphConvolution(in_channels=128,
                                       out_channels=out_channels,
                                       normalize=False,
                                       qi=False,
                                       qo=True,
                                       num_bits=bit_widths[2],
                                       )
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn_1.reset_parameters()
        self.gcn_2.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.gcn_1(x, edge_index, edge_attr)
        x = self.relu_1(x)
        x = dropout(x, p=0.5, training=self.training)
        x = self.gcn_2(x, edge_index, edge_attr)
        return x


if __name__ == '__main__':
    device_ = "cpu" if cuda.is_available() else "cpu"

    # Cora dataset settings
    n = 169_343
    e = 1_166_243
    num_feature = 128
    out_feature = 40

    # Define the model
    bit_width = [[-1, -1, -1, -1, -1, -1],
                 [-1],
                 [-1, -1, -1, -1, -1, -1],
                 ]
    model = QGCN(num_channels=num_feature, out_channels=out_feature, bit_widths=bit_width)

    # Prepare model input
    model_input = {"x": randn(n, num_feature, device=device_),
                   "edge_index": randint(n, size=(2, e), dtype=int64, device=device_),
                   "edge_attr": ones(e, device=device_),
                   }

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
        "fillcolor": "white",
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
    G.draw(png_buffer, format="png", prog="dot", args="-Gdpi=300")
    G.draw("number_of_ops_two_layer_gcn_depth_2.pdf", format="pdf", prog="dot")
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
