import argparse
from datetime import datetime
from os import makedirs, environ
from os.path import exists
from pathlib import Path

from torch_operation_counter import OperationsCounterMode


def seed_everything(seed):
    seed = 42 if not seed else seed
    import random
    random.seed(seed)
    environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from torch_geometric import seed_everything
    seed_everything(seed)


seed_everything(42)


import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Module, CrossEntropyLoss, ModuleList, BatchNorm1d
from torch import no_grad, cuda, bool, from_numpy, zeros, ones, device as torch_device

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool

from quantization.fixed_modules.parametric.graph_isomorphism import QGraphIsomorphism
from quantization.fixed_modules.parametric.linear import QLinear
from quantization.fixed_modules.non_parametric.activations import QReLU
from training.logger import setup_logger
from utility import plot_hist_statistics, plot_2d_tensor_magnitude, format_fraction
from transormed_tudataset import TransformedTUDataset


class LinearReLULinearReLU(Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 qi,
                 qo,
                 num_bits,
                 ):
        super(LinearReLULinearReLU, self).__init__()
        self.lin_1 = QLinear(in_features=in_channels,
                             out_features=hidden_channels,
                             qi=qi,
                             qo=True,
                             num_bits=num_bits,
                             is_signed=True,
                             )
        self.relu_1 = QReLU(num_bits=num_bits)
        self.lin_2 = QLinear(in_features=hidden_channels,
                             out_features=out_channels,
                             qi=False,
                             qo=qo,
                             num_bits=num_bits,
                             is_signed=False,
                             )
        self.relu_2 = QReLU(num_bits=num_bits)
        # Warning: BatchNorm1d is not quantized within this module and can't be fused with the linear layers
        self.batch_norm = BatchNorm1d(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_1.reset_parameters()
        self.relu_1.reset_parameters()
        self.lin_2.reset_parameters()
        self.relu_2.reset_parameters()
        self.batch_norm.reset_parameters()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.relu_1(x)
        x = self.lin_2(x)
        x = self.relu_2(x)
        # Warning: BatchNorm1d is not quantized within this module and can't be fused with the linear layers
        x = self.batch_norm(x)
        return x


class QGIN(Module):
    def __init__(self,
                 in_channels,
                 num_layers,
                 hidden_channels,
                 out_channels,
                 ):
        super(QGIN, self).__init__()

        self.convs = ModuleList()
        for i in range(num_layers):
            is_first_layer = i == 0
            mlp_i = LinearReLULinearReLU(in_channels if is_first_layer else hidden_channels,
                                         hidden_channels,
                                         hidden_channels,
                                         qi=False,
                                         qo=False,
                                         num_bits=[-1, -1, -1],
                                         )
            gin_i = QGraphIsomorphism(nn=mlp_i, qi=is_first_layer, qo=False, num_bits=[-1, -1])
            self.convs.append(gin_i)

        self.lin1 = QLinear(in_features=hidden_channels,
                            out_features=hidden_channels,
                            qi=False,
                            qo=False,
                            num_bits=[-1, -1, -1],
                            )
        self.relu = QReLU(num_bits=[-1])
        self.lin2 = QLinear(in_features=hidden_channels,
                            out_features=out_channels,
                            qi=False,
                            qo=False,
                            num_bits=[-1, -1, -1],
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
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
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
    parser.add_argument("--dataset_name", type=str, default="REDDIT-MULTI-5K")
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--step_size", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--plot_statistics", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")

    args = parser.parse_args()
    arguments = vars(args)

    log_dir_name = (f"{args.dataset_name}/"
                    f"batch_size_{args.batch_size}/"
                    f"hidden_{args.hidden_channels}/"
                    f"layers_{args.num_layers}/"
                    f"wd_{format_fraction(args.weight_decay)}/"
                    f"lr_{format_fraction(args.lr)}/"
                    f"bit_width_fp32"
                    )

    if not exists(args.log_dir + "/" + log_dir_name):
        makedirs(args.log_dir + "/" + log_dir_name, exist_ok=True)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name = f"{log_dir_name}/log_{current_time}"
    logger = setup_logger(filename=f"{args.log_dir}/{log_file_name}.log", verbose=True)

    [logger.info(f"{k}: {v}") for k, v in arguments.items()]
    logger.info("=" * 100)

    root = Path(__file__).resolve().parent.parent
    data_dir = root.joinpath("data")

    device = torch_device(args.device)
    dataset = TransformedTUDataset(root=data_dir, name=args.dataset_name, use_node_attr=True, use_edge_attr=True, cleaned=True)

    y = [d.y.item() for d in dataset]
    features_dim = dataset[0].x.shape[1]
    n_classes = len(np.unique(y))
    criterion = CrossEntropyLoss()
    splits = k_fold(dataset, args.num_folds)

    model = QGIN(in_channels=features_dim,
                 hidden_channels=args.hidden_channels,
                 out_channels=n_classes,
                 num_layers=args.num_layers,
                 ).to(device)

    accuracies, bit_operations = [], []

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

        logger.info(f"Fold {fold + 1}/{args.num_folds} - Accuracy: {test_accuracies[best_epoch]:.2f}%")
        accuracies += [test_accuracies[best_epoch], ]
        with no_grad():
            batch_i = next(iter(train_loader)).to(device)
            with OperationsCounterMode(model) as ops_counter:
                model(batch_i)
        bit_operations += [ops_counter.total_main_operation * 32, ]

        logger.info("_" * 100)

        # Plot the distribution of each layer's output
        if args.plot_statistics:
            save_path = root.joinpath("plots", args.dataset_name, f"fold_{fold + 1}")
            if not save_path.exists():
                save_path.mkdir(parents=True)
            extension = "png"
            show = True
            training_set_sample = next(iter(train_loader))
            x, edge_index, batch = training_set_sample.x, training_set_sample.edge_index, training_set_sample.batch
            x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)
            for i, conv in enumerate(model.convs):
                x = conv.mp(edge_index, x=x)
                hist_plot_dir = save_path.joinpath(f"distribution_layer_{i}_after_mp.{extension}")
                tensor_plot_dir = save_path.joinpath(f"tensor_values_layer_{i}_after_mp.{extension}")
                plot_hist_statistics(x, f"Fold {fold + 1} / {args.num_folds} - Distribution after the message passing at layer {i}", show=show, save_path=hist_plot_dir)
                plot_2d_tensor_magnitude(x, f"Fold {fold + 1} / {args.num_folds} - Tensor values after the message passing at layer {i}", show=show, save_path=tensor_plot_dir)
                x = conv.nn(x)
                hist_plot_dir = save_path.joinpath(f"distribution_layer_{i}_after_nn.{extension}")
                tensor_plot_dir = save_path.joinpath(f"tensor_values_layer_{i}_after_nn.{extension}")
                plot_hist_statistics(x, f"Fold {fold + 1} / {args.num_folds} - Distribution after the MLP at layer {i} (Activation)", show=show, save_path=hist_plot_dir)
                plot_2d_tensor_magnitude(x, f"Fold {fold + 1} / {args.num_folds} - Tensor values after the MLP at layer {i} (Activation)", show=show, save_path=tensor_plot_dir)
            x = global_max_pool(x, batch)
            hist_plot_dir = save_path.joinpath(f"distribution_layer_{i + 1}_after_global_max_pooling.{extension}")
            tensor_plot_dir = save_path.joinpath(f"tensor_values_layer_{i + 1}_after_global_max_pooling.{extension}")
            plot_hist_statistics(x, f"Fold {fold + 1} / {args.num_folds} - Distribution after the global max pooling", show=show, save_path=hist_plot_dir)
            plot_2d_tensor_magnitude(x, f"Fold {fold + 1} / {args.num_folds} - Tensor values after the global max pooling", xaxis_title="Graph", yaxis_title="Aggregated features", show=show, save_path=tensor_plot_dir)
            hist_plot_dir = save_path.joinpath(f"distribution_layer_{i + 2}_after_first_linear_layer.{extension}")
            tensor_plot_dir = save_path.joinpath(f"tensor_values_layer_{i + 2}_after_first_linear_layer.{extension}")
            x = model.lin1(x)
            plot_hist_statistics(x, f"Fold {fold + 1} / {args.num_folds} - Distribution after the first linear layer", show=show, save_path=hist_plot_dir)
            plot_2d_tensor_magnitude(x, f"Fold {fold + 1} / {args.num_folds} - Tensor values after the first linear layer", xaxis_title="Graph", show=show, save_path=tensor_plot_dir)
            x = model.relu(x)
            hist_plot_dir = save_path.joinpath(f"distribution_layer_{i + 3}_after_relu_activation.{extension}")
            tensor_plot_dir = save_path.joinpath(f"tensor_values_layer_{i + 3}_after_relu_activation.{extension}")
            plot_hist_statistics(x, f"Fold {fold + 1} / {args.num_folds} - Distribution after the ReLU activation", show=show, save_path=hist_plot_dir)
            plot_2d_tensor_magnitude(x, f"Fold {fold + 1} / {args.num_folds} - Tensor values after the ReLU activation", xaxis_title="Graph", show=show, save_path=tensor_plot_dir)
            x = model.lin2(x)
            hist_plot_dir = save_path.joinpath(f"distribution_layer_{i + 4}_after_second_linear_layer.{extension}")
            tensor_plot_dir = save_path.joinpath(f"tensor_values_layer_{i + 4}_after_second_linear_layer.{extension}")
            plot_hist_statistics(x, f"Fold {fold + 1} / {args.num_folds} - Distribution after the second linear layer", show=show, save_path=hist_plot_dir)
            plot_2d_tensor_magnitude(x, f"Fold {fold + 1} / {args.num_folds} - Tensor values after the second linear layer", xaxis_title="Graph", show=show, save_path=tensor_plot_dir)

    logger.info(f"Results for {args.num_folds}-fold cross-validation:")
    logger.info(f"Accuracies: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    logger.info(f"Bit operations: {np.mean(bit_operations):.2f} ± {np.std(bit_operations):.2f}")
    logger.info("=" * 100)
