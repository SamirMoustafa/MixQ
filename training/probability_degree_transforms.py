# Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, 2020
"""
By convention, masks should have true elements at positions where higher precision should be used
"""

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import degree


class ProbabilisticHighDegreeMask:
    def __init__(self, low_probability, high_probability, per_graph=True):
        self.low_prob = low_probability
        self.high_prob = high_probability
        self.process_per_graph = per_graph

    def _process_graph(self, graph):
        # Note that:
        # 1. The probability of being protected increases as the in-degree increases
        # 2. All nodes with the same in-degree have the same bernoulli p
        # 3. You can set this such that all nodes have some probability of being non-protected

        n = graph.num_nodes
        in_degree = degree(graph.edge_index[1], n, dtype=torch.long)
        counts = torch.bincount(in_degree)

        step_size = (self.high_prob - self.low_prob) / n
        in_degree_probabilities = counts * step_size
        in_degree_probabilities = torch.cumsum(in_degree_probabilities, dim=0)
        in_degree_probabilities += self.low_prob
        graph.probability_mask = in_degree_probabilities[in_degree]

        return graph

    def __call__(self, data):
        if self.process_per_graph and isinstance(data, Batch):
            graphs = data.to_data_list()
            processed_graphs_list = [self._process_graph(graph) for graph in graphs]
            return Batch.from_data_list(processed_graphs_list)
        else:
            return self._process_graph(data)
