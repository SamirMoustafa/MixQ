import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from torch import tensor, zeros
from torch.testing import assert_close
from torch_geometric.nn import SimpleConv

from quantization.fixed_modules.non_parametric.message_passing import QMessagePassing
from utils import generate_erdos_renyi_graph


def scatter_add(src, index):
    num_indices = index.max() + 1
    num_features = src.shape[1]
    x = zeros((num_indices, num_features), dtype=src.dtype)
    x.index_add_(0, index, src)
    return x


class TestMessagePassing(TestCase):
    def setUp(self):
        # Initialize the QuantizableMessagePassing object
        self.qmp = QMessagePassing(False, False, [8, 8], True, "tensor")

        # Mock the message and aggregate methods
        message_func = lambda x_j, edge_weight: x_j
        aggregate_func = lambda x, index, dim_size: scatter_add(x, index)
        self.qmp.message = MagicMock(side_effect=message_func)
        self.qmp.aggregate = MagicMock(side_effect=aggregate_func)

    def test_cycle_message_passing(self):
        # Define a simple graph
        edge_index = tensor([[0, 1, 2],
                             [1, 2, 0]])  # 3 edges forming a cycle
        x = tensor([[1.], [2.], [3.]])  # Node features

        # Expected output: sum of each node's neighbors' features
        expected_x = x.clone()  # Copy the node features, as the original tensor will be modified

        # Perform message passing 3 times such that each node receives messages from its neighbors 3 times
        # Since the edges form a cycle, the node features should be the same after 3 iterations
        x = self.qmp(edge_index, x=x)
        x = self.qmp(edge_index, x=x)
        x = self.qmp(edge_index, x=x)

        # Verify the message and aggregate methods were called correctly
        self.qmp.message.assert_called()
        self.qmp.aggregate.assert_called()

        # Assert the result is as expected
        assert_close(x, expected_x, rtol=1e-5, atol=1e-5,
                     msg="The message passing output is incorrect, expected: {}, got: {}".format(expected_x, x))

    def test_random_message_passing(self):
        # Define a simple graph
        edge_index, x, _ = generate_erdos_renyi_graph(50, 0.5, 100, 1)
        expected_x = SimpleConv()(edge_index=edge_index, x=x)
        x = self.qmp(edge_index, x=x)

        # Verify the message and aggregate methods were called correctly
        self.qmp.message.assert_called()
        self.qmp.aggregate.assert_called()

        # Assert the result is as expected
        assert_close(x, expected_x, rtol=1e-5, atol=1e-5,
                     msg="The message passing output is incorrect, expected: {}, got: {}".format(expected_x, x))


if __name__ == '__main__':
    unittest.main()
