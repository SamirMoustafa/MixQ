from operator import itemgetter

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch import LongTensor


def prepare_loaders(dataset, splits, batch_size=32, index=0):
    train_idx = LongTensor(splits["train"][index])
    val_idx = LongTensor(splits["val"][index])
    test_idx = LongTensor(splits["test"][index])

    train_dataset = itemgetter(*train_idx)(dataset)
    val_dataset = itemgetter(*val_idx)(dataset)
    test_dataset = itemgetter(*test_idx)(dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


class Dataset(InMemoryDataset):
    def __init__(self, data_list):
        super(Dataset, self).__init__("./temp")
        self.data, self.slices = self.collate(data_list)

    def _process(self):
        pass
    def _download(self):
        pass
