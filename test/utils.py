from torch import randint, float32
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops


def generate_erdos_renyi_graph(num_nodes, p_edges, num_features, num_classes, dtype=float32):
    """
    Generate a random graph with an Erdos-Renyi model

    :param num_nodes: Number of nodes
    :param p_edges: Probability of an edge between two nodes
    :param num_features: Length of the feature vector
    :param num_classes: Number of classes
    :param dtype: Data type of the features
    :return: Tuple of edge_index, features, labels
    """
    edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=False)
    edge_index, _ = remove_self_loops(edge_index)
    features = randint(0, 10, (num_nodes, num_features), dtype=dtype)
    labels = randint(0, num_classes, (num_nodes,))
    return edge_index, features, labels
