import unittest

from torch import randint, float32, cuda, no_grad, testing
from torch.nn import Module, ModuleList
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from quantization.fixed_modules.parametric import QGraphConvolution


class QGraphConvolutionalLayer(QGraphConvolution):

    def set_forward_function(self, forward_function):
        self.forward = forward_function

    def quantize_inference(self, x, edge_index, size=None):
        qx = self.lin.qi.quantize(x)
        qx = super().quantize_inference(qx, edge_index, size)
        x = self.mp.qo.dequantize(qx)
        return x


class QGraphConvolutionalNetwork(Module):
        def __init__(self, num_features, num_classes, num_layers, num_bits, is_signed):
            super(QGraphConvolutionalNetwork, self).__init__()
            self.num_features = num_features
            self.num_classes = num_classes
            self.num_layers = num_layers

            self.layers = ModuleList()
            for i in range(num_layers):
                is_first_layer = i == 0
                is_last_layer = i == num_layers - 1
                in_channels = num_features if is_first_layer else num_classes
                out_channels = num_classes if is_last_layer else num_classes
                layer = QGraphConvolution(in_channels=in_channels,
                                          out_channels=out_channels,
                                          qi=is_first_layer,
                                          qo=True,
                                          num_bits=num_bits[i],
                                          is_signed=is_signed,
                                          )
                self.layers += [layer]

        def set_forward_function(self, forward_function):
            self.forward = forward_function

        def forward(self, x, edge_index):
            for layer in self.layers:
                x = layer(x, edge_index)
            return x

        def simulated_quantize_forward(self, x, edge_index):
            for i, layer in enumerate(self.layers):
                x = layer.simulated_quantize_forward(x, edge_index)
            return x

        def freeze(self):
            self.layers[0].freeze()
            qo = self.layers[0].qo
            for i, layer in enumerate(self.layers[1:]):
                layer.freeze(qi=qo)
                qo = layer.qo

        def quantize_inference(self, x, edge_index):
            qx = self.layers[0].qi.quantize(x)
            for i, layer in enumerate(self.layers):
                qx = layer.quantize_inference(qx, edge_index)
            x = self.layers[-1].qo.dequantize(qx)
            return x


def generate_erdos_renyi_graph(num_nodes, p_edges, num_features, num_classes):
    edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=False)
    edge_index, _ = remove_self_loops(edge_index)
    features = randint(0, 10, (num_nodes, num_features), dtype=float32)
    labels = randint(0, num_classes, (num_nodes,))
    return edge_index, features, labels


class BaseTest:
    def setUp(self):
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.num_nodes = 50
        self.p_edges = 0.25
        self.num_features = 3
        self.num_classes = 2

        self.bit = 8

        edge_index, features, labels = generate_erdos_renyi_graph(self.num_nodes,
                                                                  self.p_edges,
                                                                  self.num_features,
                                                                  self.num_classes,
                                                                  )
        self.x = features.to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.y = labels.to(self.device)

    def test_forward_functions(self):
        if not hasattr(self, "model"):
            raise NotImplementedError("Model must be defined. If you are testing a base class, do not run this test."
                                      "Since it is not meant to be run on its own.")

        self.model = self.model.to(self.device)
        self.model.train()

        self.model.set_forward_function(self.model.forward)
        full_precision_output = self.model(self.x, self.edge_index)

        self.model.set_forward_function(self.model.simulated_quantize_forward)
        simulated_quantization_output = self.model(self.x, self.edge_index)

        self.model.eval()
        self.model.freeze()
        self.model.set_forward_function(self.model.quantize_inference)
        with no_grad():
            quantized_output = self.model(self.x, self.edge_index)

        self.assertEqual(full_precision_output.shape, simulated_quantization_output.shape, "Output shapes must match.")
        self.assertEqual(full_precision_output.shape, quantized_output.shape, "Output shapes must match.")

        testing.assert_close(simulated_quantization_output, quantized_output, rtol=1e-2, atol=1e-2)


class TestGraphConvolutionalLayer(BaseTest, unittest.TestCase):
    def setUp(self):
        super().setUp()
        num_bits = [self.bit, self.bit, self.bit, self.bit, self.bit, self.bit]
        self.model = QGraphConvolutionalLayer(in_channels=self.num_features,
                                              out_channels=self.num_classes,
                                              qi=True,
                                              qo=True,
                                              num_bits=num_bits,
                                              is_signed=True,
                                              )


class TestGraphConvolutionalNetwork(BaseTest, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.num_layers = 5
        num_bits = [[self.bit, self.bit, self.bit, self.bit, self.bit, self.bit]
                    for _ in range(self.num_layers)]
        self.model = QGraphConvolutionalNetwork(num_features=self.num_features,
                                                num_classes=self.num_classes,
                                                num_layers=self.num_layers,
                                                num_bits=num_bits,
                                                is_signed=True,
                                                )


if __name__ == '__main__':
    unittest.main()
