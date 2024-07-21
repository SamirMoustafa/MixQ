from torch import float32, cat, long
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree, Compose
from torch_geometric.utils import degree


def compute_mean_and_std_degrees(dataset):
    degrees = []
    for data in dataset:
        degrees += [degree(data.edge_index[0], dtype=float32)]
    return cat(degrees).mean().item(), cat(degrees).std().item()


def compute_max_degree(dataset):
    max_degree = 0
    degrees = []
    for data in dataset:
        degrees += [degree(data.edge_index[0], dtype=long)]
        max_degree = max(max_degree, degrees[-1].max().item())
    return max_degree


class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=float32)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class TransformedTUDataset(TUDataset):
    def __init__(self,
                 root: str,
                 name: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 force_reload: bool = False,
                 use_node_attr: bool = False,
                 use_edge_attr: bool = False,
                 cleaned: bool = False):

        super(TransformedTUDataset, self).__init__(root=root,
                                                   name=name,
                                                   transform=transform,
                                                   pre_transform=pre_transform,
                                                   pre_filter=pre_filter,
                                                   force_reload=force_reload,
                                                   use_node_attr=use_node_attr,
                                                   use_edge_attr=use_edge_attr,
                                                   cleaned=cleaned)

        if self.data.x is None:
            max_degree = compute_max_degree(self)
            if max_degree < 1000:
                degree_transform = OneHotDegree(max_degree)
            else:
                degree_mean, degree_std = compute_mean_and_std_degrees(self)
                degree_transform = NormalizedDegree(degree_mean, degree_std)

            if self.transform is not None:
                self.transform = Compose([degree_transform, self.transform])
            else:
                self.transform = degree_transform
