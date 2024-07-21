import time
import os
import csv
from pathlib import Path
from pickle import load as pickle_load

import numpy as np
import networkx as nx
from torch import load as torch_load, unique, tensor, float32, int64
from torch_geometric.data import Data, InMemoryDataset


class CSL(InMemoryDataset):
    """
        Circular Skip Link Graphs: 
        Source: https://github.com/PurdueMINDS/RelationalPooling/
    """

    def __init__(self, root=Path(__file__).resolve().parents[2].joinpath('data', 'CSL')):
        super().__init__(root)
        self.name = "CSL"
        self.adj_list = pickle_load(open(os.path.join(root, 'graphs_Kary_Deterministic_Graphs.pkl'), 'rb'))
        self.graph_labels = torch_load(os.path.join(root, 'y_Kary_Deterministic_Graphs.pt'))
        self.graph_lists = []
        self.n_classes = len(unique(self.graph_labels))
        self.n_samples = len(self.graph_labels)
        self.num_node_type = 1  # 41
        self.num_edge_type = 1  # 164
        self.n_splits = 5
        self.root_dir = root
        self._prepare()

    def _prepare(self):
        t0 = time.time()
        graph_list = []
        feature_list = []
        print("[I] Preparing Circular Skip Link Graphs v4 ...")
        for idx in range(self.n_samples):
            G = nx.from_scipy_sparse_array(self.adj_list[idx]).to_directed()
            G.remove_edges_from(nx.selfloop_edges(G))
            X = np.ones((len(G), 1))
            graph_list.append(G)
            feature_list.append(X)

        assert len(graph_list) == len(feature_list) == len(self.graph_labels), "Lengths do not match"
        self.dataset = []
        for i in range(len(graph_list)):
            edge_index = tensor([*graph_list[i].edges], dtype=int64).t()
            x = tensor(feature_list[i], dtype=float32)
            y = tensor([self.graph_labels[i]], dtype=int64)
            data_i = Data(x=x, edge_index=edge_index, y=y)
            self.dataset.append(data_i)

        print("[I] Finished preparation after {:.4f}s".format(time.time() - t0))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_all_splits_idx(self):
        """
            - Split total number of graphs into 3 (train, val and test) in 3:1:1
            - Stratified split proportionate to the original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
            - Preparing 5 such combinations of indexes split to be used in Graph NNs
            - As with KFold, each of the 5-fold have unique test set.
        """

        all_idx = {}

        # reading idx from the files
        for section in ['train', 'val', 'test']:
            with open(self.root_dir / (self.name + '_' + section + '.index'), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        return all_idx

    @property
    def num_classes(self):
        return len(unique(self.graph_labels))


if __name__ == "__main__":
    dataset = CSL()
    print(dataset[0])
    print(dataset.get_all_splits_idx())
    print(dataset.num_classes)
