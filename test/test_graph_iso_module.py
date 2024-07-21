import unittest
from copy import deepcopy

from torch import randint, float32, cuda, no_grad, testing, zeros, rand
from torch.nn import Module, ModuleList
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from quantization.fixed_modules.parametric import QGraphIsomorphism, MaskQuantGraphIsomorphism,  QLinear


class QGraphIsomorphismLayer(QGraphIsomorphism):

    def set_forward_function(self, forward_function):
        self.forward = forward_function

    def quantize_inference(self, x, edge_index, size=None):
        qx = self.mp.qi.quantize(x)
        qx = super().quantize_inference(qx, edge_index, size)
        x = self.nn.qo.dequantize(qx)
        return x


class MaskQuantGraphIsomorphismLayer(MaskQuantGraphIsomorphism):

    def set_forward_function(self, forward_function):
        self.forward = forward_function

    def quantize_inference(self, x, edge_index, size=None):
        qx = self.mp.qi.quantize(x)
        qx = super().quantize_inference(qx, edge_index, size)
        x = self.nn.qo.dequantize(qx)
        return x


class QGraphIsomorphismNetwork(Module):
    def __init__(self, mlp, num_layers, num_bits, is_signed):
        super(QGraphIsomorphismNetwork, self).__init__()
        self.num_layers = num_layers

        self.layers = ModuleList()
        for i in range(num_layers):
            is_first_layer = i == 0
            mlp = deepcopy(mlp)
            layer = QGraphIsomorphism(nn=mlp,
                                      qi=is_first_layer,
                                      qo=True,
                                      num_bits=num_bits,
                                      is_signed=is_signed,
                                      )
            self.layers += [layer]

    def set_forward_function(self, forward_function):
        self.forward = forward_function
        for layer in self.layers:
            forward_func = getattr(layer.nn, forward_function.__name__)
            layer.nn.forward = forward_func

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
    features = rand(num_nodes, num_features, dtype=float32)
    labels = randint(0, num_classes, (num_nodes,))
    return edge_index, features, labels


class BaseTest:
    def setUp(self):
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.num_nodes = 50
        self.p_edges = 0.25
        self.num_features = 3

        self.bit = 16  # If the quantization bit is changed, to be 8 the test will be unstable.

        edge_index, features, labels = generate_erdos_renyi_graph(self.num_nodes,
                                                                  self.p_edges,
                                                                  self.num_features,
                                                                  self.num_features,
                                                                  )
        self.x = features.to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.y = labels.to(self.device)

        self.args_for_simulated_quantization = {}

        self.mlp = QLinear(in_features=self.num_features,
                           out_features=self.num_features,
                           qi=False,
                           qo=True,
                           num_bits=[self.bit, self.bit, self.bit],
                           is_signed=True,
                           )

    def test_forward_functions(self):
        if not hasattr(self, "model"):
            raise NotImplementedError("Model must be defined. If you are testing a base class, do not run this test."
                                      "Since it is not meant to be run on its own.")

        self.model = self.model.to(self.device)
        self.model.train()

        self.model.set_forward_function(self.model.forward)
        full_precision_output = self.model(self.x, self.edge_index)

        self.model.set_forward_function(self.model.simulated_quantize_forward)
        simulated_quantization_output = self.model(self.x, self.edge_index, **self.args_for_simulated_quantization)

        self.model.eval()
        self.model.freeze()
        self.model.set_forward_function(self.model.quantize_inference)
        with no_grad():
            quantized_output = self.model(self.x, self.edge_index)

        self.assertEqual(full_precision_output.shape, simulated_quantization_output.shape, "Output shapes must match.")
        self.assertEqual(full_precision_output.shape, quantized_output.shape, "Output shapes must match.")

        testing.assert_close(simulated_quantization_output, quantized_output, rtol=1e-2, atol=1e-2)


class TestGraphIsomorphismLayer(BaseTest, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model = QGraphIsomorphismLayer(nn=self.mlp,
                                            qi=True,
                                            qo=True,
                                            num_bits=[self.bit, self.bit],
                                            is_signed=True,
                                            )


class TestMaskQuantGraphIsomorphismLayer(BaseTest, unittest.TestCase):
    def setUp(self):
        super().setUp()
        false_mask = zeros((self.edge_index.max().item() + 1,), device=self.device, dtype=bool)
        self.args_for_simulated_quantization.update({"binary_mask": false_mask})
        self.model = MaskQuantGraphIsomorphismLayer(nn=self.mlp,
                                                    qi=True,
                                                    qo=True,
                                                    num_bits=[self.bit, self.bit],
                                                    is_signed=True,
                                                    quant_percentile=0.001,
                                                    )


# This test is not stable. It sometimes fails due to the random nature of the graph generation.
class TestGraphIsomorphismNetworkWithQLinear(BaseTest, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.num_layers = 5
        self.model = QGraphIsomorphismNetwork(mlp=self.mlp,
                                              num_layers=self.num_layers,
                                              num_bits=[self.bit, self.bit],
                                              is_signed=True,
                                              )


if __name__ == '__main__':
    unittest.main()
