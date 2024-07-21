# Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, 2020
from pathlib import Path

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Module, CrossEntropyLoss, ModuleList, BatchNorm1d
from torch import no_grad, cuda, bool, from_numpy, zeros, ones, bernoulli, device as torch_device


from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree, Compose

from training.probability_degree_transforms import ProbabilisticHighDegreeMask
from quantization.fixed_modules.parametric.graph_isomorphism import MaskQuantGraphIsomorphismWithSkipConnections
from quantization.fixed_modules.parametric.linear import QMinMaxRangesLinear, QMinMaxRangesLinearBatchNormReLU
from quantization.fixed_modules.non_parametric.activations import QReLU
from transormed_tudataset import compute_mean_and_std_degrees, compute_max_degree, NormalizedDegree, \
    TransformedTUDataset


class LinearReLULinearReLU(Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 qi,
                 qo,
                 bit_widths,
                 use_momentum,
                 momentum,
                 percentile,
                 sample_ratio,
                 ):
        super(LinearReLULinearReLU, self).__init__()
        self.lin_1 = QMinMaxRangesLinear(in_features=in_channels,
                                         out_features=hidden_channels,
                                         qi=qi,
                                         qo=True,
                                         num_bits=bit_widths[0],
                                         is_signed=True,
                                         quant_percentile=percentile,
                                         quant_use_momentum=use_momentum,
                                         quant_momentum=momentum,
                                         quant_sample_ratio=sample_ratio,
                                         )
        self.relu_1 = QReLU(num_bits=bit_widths[1])
        self.lin_2 = QMinMaxRangesLinear(in_features=hidden_channels,
                                         out_features=out_channels,
                                         qi=True,
                                         qo=qo,
                                         num_bits=bit_widths[2],
                                         is_signed=False,
                                         quant_percentile=percentile,
                                         quant_use_momentum=use_momentum,
                                         quant_momentum=momentum,
                                         quant_sample_ratio=sample_ratio,
                                         )
        self.relu_2 = QReLU(num_bits=bit_widths[3])
        # Warning: BatchNorm1d is not quantized within this module and can't be fused with the linear layers
        self.batch_norm = BatchNorm1d(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_1.reset_parameters()
        self.relu_1.reset_parameters()
        self.lin_2.reset_parameters()
        self.relu_2.reset_parameters()
        self.batch_norm.reset_parameters()

    def simulated_quantize_forward(self, x):
        x = self.lin_1.simulated_quantize_forward(x)
        x = self.relu_1.simulated_quantize_forward(x)
        x = self.lin_2.simulated_quantize_forward(x)
        x = self.relu_2.simulated_quantize_forward(x)
        # Warning: BatchNorm1d is not quantized within this module and can't be fused with the linear layers
        x = self.batch_norm(x)
        return x


class QGIN(Module):
    def __init__(self,
                 in_channels,
                 num_layers,
                 hidden_channels,
                 out_channels,
                 num_bits,
                 use_momentum,
                 momentum,
                 percentile,
                 sample_ratio,
                 ):
        super(QGIN, self).__init__()

        self.convs = ModuleList()
        for i in range(num_layers):
            is_first_layer = i == 0
            mlp_i = LinearReLULinearReLU(in_channels if is_first_layer else hidden_channels,
                                         hidden_channels,
                                         hidden_channels,
                                         qi=True,
                                         qo=True,
                                         bit_widths=num_bits[i][0],
                                         use_momentum=use_momentum,
                                         momentum=momentum,
                                         percentile=percentile,
                                         sample_ratio=sample_ratio,
                                         )
            gin_i = MaskQuantGraphIsomorphismWithSkipConnections(nn=mlp_i,
                                                                 qi=is_first_layer,
                                                                 qo=True,
                                                                 num_bits=num_bits[i][1],
                                                                 is_signed=True,
                                                                 quant_use_momentum=use_momentum,
                                                                 quant_momentum=momentum,
                                                                 quant_percentile=percentile,
                                                                 quant_sample_ratio=sample_ratio,
                                                                 )
            self.convs.append(gin_i)

        self.lin1 = QMinMaxRangesLinear(in_features=hidden_channels,
                                        out_features=hidden_channels,
                                        qi=True,
                                        qo=True,
                                        num_bits=num_bits[num_layers],
                                        is_signed=False,
                                        quant_percentile=percentile,
                                        quant_use_momentum=use_momentum,
                                        quant_momentum=momentum,
                                        quant_sample_ratio=sample_ratio,
                                        )
        self.relu = QReLU(num_bits=[num_layers + 1])
        self.lin2 = QMinMaxRangesLinear(in_features=hidden_channels,
                                        out_features=out_channels,
                                        qi=True,
                                        qo=True,
                                        num_bits=num_bits[num_layers + 2],
                                        is_signed=False,
                                        quant_percentile=percentile,
                                        quant_use_momentum=use_momentum,
                                        quant_momentum=momentum,
                                        quant_sample_ratio=sample_ratio,
                                        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.relu.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        binary_mask = bernoulli(data.probability_mask).to(bool)

        for i, conv in enumerate(self.convs):
            x = conv.simulated_quantize_forward(x, edge_index, binary_mask)
        x = global_max_pool(x, batch)
        x = self.lin1.simulated_quantize_forward(x)
        x = self.relu.simulated_quantize_forward(x)
        x = self.lin2.simulated_quantize_forward(x)
        return x


def train(model, criterion, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)


@no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset) * 100


@no_grad()
def evaluate_loss(model, criterion, loader, device):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss += criterion(out, data.y.view(-1)).item() * data.num_graphs
    return loss / len(loader.dataset)


def get_train_val_test_loaders(dataset, train_index, val_index, test_index):
    train_loader = DataLoader(dataset[train_index], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_index], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[test_index], batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    test_indices, train_indices = [], []
    for _, idx in skf.split(zeros(len(dataset)), dataset.data.y):
        test_indices.append(from_numpy(idx))
    val_indices = [test_indices[i - 1] for i in range(folds)]
    for i in range(folds):
        train_mask = ones(len(dataset), dtype=bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))
    return train_indices, val_indices, test_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="REDDIT-BINARY")
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")

    parser.add_argument("--num_bits", type=int, default=8)
    parser.add_argument("--mask_low_probability", type=float, default=0.0)
    parser.add_argument("--mask_high_probability", type=float, default=0.1)
    parser.add_argument("--quant_percentile", type=float, default=0.001)
    parser.add_argument("--quant_use_momentum", action="store_true")
    parser.add_argument("--quant_momentum", type=float, default=0.01)
    parser.add_argument("--quant_sample_ratio", type=float, default=None)
    args = parser.parse_args()
    arguments = vars(args)
    [print(f"{k}: {v}") for k, v in arguments.items()]

    root = Path(__file__).resolve().parent.parent
    data_dir = root.joinpath("data")

    device = torch_device(args.device)
    dataset = TransformedTUDataset(root=data_dir, name=args.dataset_name, use_node_attr=True, use_edge_attr=True, cleaned=True)

    high_probability = min(args.mask_low_probability + args.mask_high_probability, 1.0)
    mask_transform = ProbabilisticHighDegreeMask(args.mask_low_probability, high_probability)
    if dataset.transform is None:
        dataset.transform = mask_transform
    else:
        dataset.transform = Compose([dataset.transform, mask_transform])

    y = [d.y.item() for d in dataset]
    features_dim = dataset[0].x.shape[1]
    n_classes = len(np.unique(y))
    criterion = CrossEntropyLoss()
    splits = k_fold(dataset, args.num_folds)

    bit_widths = [[[[args.num_bits, args.num_bits, args.num_bits],
                    [args.num_bits],
                    [args.num_bits, args.num_bits, args.num_bits],
                    [args.num_bits]
                    ],  # MLP Bit-widths
                   [args.num_bits, args.num_bits]  # Graph Isomorphism Bit-widths
                   ] for _ in range(args.num_layers)]
    bit_widths += [[args.num_bits, args.num_bits, args.num_bits],
                   [args.num_bits],
                   [args.num_bits, args.num_bits, args.num_bits],
                   ]
    model = QGIN(in_channels=features_dim,
                 hidden_channels=args.hidden_channels,
                 out_channels=n_classes,
                 num_layers=args.num_layers,
                 num_bits=bit_widths,
                 percentile=args.quant_percentile,
                 use_momentum=args.quant_use_momentum,
                 momentum=args.quant_momentum,
                 sample_ratio=args.quant_sample_ratio,
                 ).to(device)

    accuracies = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(zip(*splits)):
        model.reset_parameters()

        train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset, train_idx, val_idx, test_idx)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        best_epoch, test_accuracies, best_val_loss = 0, [], float("inf")
        pbar = tqdm(range(args.epochs), desc="Epoch 0 Loss 0")
        for epoch in pbar:
            train_loss = train(model, criterion, optimizer, train_loader)
            val_loss = evaluate_loss(model, criterion, val_loader, device)
            test_accuracies += [evaluate_accuracy(model, test_loader, device), ]
            scheduler.step()
            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
            pbar.set_description(f"Epoch: {epoch}, Train Loss: {train_loss:.2f},  "
                                 f"Val Loss: {val_loss:.2f}, "
                                 f"Best Acc: {test_accuracies[best_epoch]:.2f}")

        print(f"Fold {fold + 1}/{args.num_folds} - Accuracy: {test_accuracies[best_epoch]:.2f}%")
        accuracies += [test_accuracies[best_epoch], ]

        print("_" * 100)

    print(f"Results for {args.num_folds}-fold cross-validation:")
    print(f"Accuracies: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    print("=" * 100)
