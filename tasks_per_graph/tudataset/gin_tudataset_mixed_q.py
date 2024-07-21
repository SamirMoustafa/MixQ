import argparse
from copy import deepcopy
from datetime import datetime
from os import makedirs, environ
from os.path import exists
from pathlib import Path


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
from torch import no_grad, cuda, bool, from_numpy, zeros, ones, device as torch_device, tensor

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool

from quantization.fixed_modules.parametric.graph_isomorphism import QGraphIsomorphism
from quantization.fixed_modules.non_parametric.activations import QReLU
from quantization.fixed_modules.parametric.linear import QLinear

from quantization.mixed_modules.parametric.graph_isomorphism import MQGraphIsomorphism
from quantization.mixed_modules.non_parametric.activations import MQReLU
from quantization.mixed_modules.parametric.linear import MQLinear
from training.logger import setup_logger
from utility import flatten_list, nested_median, nested_std, format_fraction, write_to_csv
from transormed_tudataset import TransformedTUDataset


class MQLinearReLULinearReLU(Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 qi,
                 qo,
                 num_bits_list,
                 quantize_per="column",
                 ):
        super(MQLinearReLULinearReLU, self).__init__()
        self.lin_1 = MQLinear(in_features=in_channels,
                              out_features=hidden_channels,
                              qi=qi,
                              qo=True,
                              num_bits_list=num_bits_list,
                              is_signed=True,
                              quantize_per=quantize_per,
                              )
        self.relu_1 = MQReLU(num_bits_list=num_bits_list, quantize_per=quantize_per, )
        self.lin_2 = MQLinear(in_features=hidden_channels,
                              out_features=out_channels,
                              qi=False,
                              qo=qo,
                              num_bits_list=num_bits_list,
                              is_signed=False,
                              quantize_per=quantize_per,
                              )
        self.relu_2 = MQReLU(num_bits_list=num_bits_list, quantize_per=quantize_per, )
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

    def calculate_weighted_loss(self, x):
        loss = self.lin_1.calculate_weighted_loss(x)
        x = self.lin_1(x)
        loss += self.relu_1.calculate_weighted_loss(x)
        x = self.relu_1(x)
        loss += self.lin_2.calculate_weighted_loss(x)
        x = self.lin_2(x)
        loss += self.relu_2.calculate_weighted_loss(x)
        return loss

    def select_top_k_winners(self, k):
        lin_1_top_1 = flatten_list(self.lin_1.select_top_k_winners(k).values())
        relu_1_top_1 = flatten_list(self.relu_1.select_top_k_winners(k).values())
        lin_2_top_1 = flatten_list(self.lin_2.select_top_k_winners(k).values())
        relu_2_top_1 = flatten_list(self.relu_2.select_top_k_winners(k).values())
        return {"mlp": [lin_1_top_1, ] + [relu_1_top_1, ] + [lin_2_top_1, ] + [relu_2_top_1, ]}

    def estimated_bit_operation_precision(self, x):
        bit_operations = 0
        bit_operations += self.lin_1.estimated_bit_operation_precision(x)
        x = self.lin_1(x)
        bit_operations += self.relu_1.estimated_bit_operation_precision(x)
        x = self.relu_1(x)
        bit_operations += self.lin_2.estimated_bit_operation_precision(x)
        x = self.lin_2(x)
        bit_operations += self.relu_2.estimated_bit_operation_precision(x)
        return bit_operations


class QLinearReLULinearReLU(Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 qi,
                 qo,
                 bit_widths,
                 quantize_per="column",
                 ):
        super(QLinearReLULinearReLU, self).__init__()
        self.lin_1 = QLinear(in_features=in_channels,
                             out_features=hidden_channels,
                             qi=qi,
                             qo=True,
                             num_bits=bit_widths[0],
                             is_signed=True,
                             quantize_per=quantize_per,
                             )
        self.relu_1 = QReLU(num_bits=bit_widths[1], quantize_per=quantize_per, )
        self.lin_2 = QLinear(in_features=hidden_channels,
                             out_features=out_channels,
                             qi=False,
                             qo=qo,
                             num_bits=bit_widths[2],
                             is_signed=False,
                             quantize_per=quantize_per,
                             )
        self.relu_2 = QReLU(num_bits=bit_widths[3], quantize_per=quantize_per, )
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


class MQGIN(Module):
    def __init__(self,
                 in_channels,
                 num_layers,
                 hidden_channels,
                 out_channels,
                 num_bits_list,
                 ):
        super(MQGIN, self).__init__()

        self.convs = ModuleList()
        for i in range(num_layers):
            is_first_layer = i == 0
            mlp_i = MQLinearReLULinearReLU(in_channels if is_first_layer else hidden_channels,
                                           hidden_channels,
                                           hidden_channels,
                                           qi=False,
                                           qo=True,
                                           num_bits_list=num_bits_list,
                                           quantize_per="column",
                                           )
            gin_i = MQGraphIsomorphism(nn=mlp_i,
                                       qi=is_first_layer,
                                       qo=True,
                                       num_bits_list=num_bits_list,
                                       is_signed=True,
                                       quantize_per="column",
                                       )
            self.convs.append(gin_i)

        self.lin1 = MQLinear(in_features=hidden_channels,
                             out_features=hidden_channels,
                             qi=True,
                             qo=True,
                             num_bits_list=num_bits_list,
                             is_signed=False,
                             quantize_per="column",
                             )
        self.relu = MQReLU(num_bits_list=num_bits_list, quantize_per="column", )
        self.lin2 = MQLinear(in_features=hidden_channels,
                             out_features=out_channels,
                             qi=True,
                             qo=True,
                             num_bits_list=num_bits_list,
                             is_signed=False,
                             quantize_per="column",
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

    def calculate_weighted_loss(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        loss = 0
        for i, conv in enumerate(self.convs):
            loss += conv.calculate_weighted_loss(x, edge_index)
            x = conv(x, edge_index)
        x = global_max_pool(x, batch)
        loss += self.lin1.calculate_weighted_loss(x)
        x = self.lin1(x)
        loss += self.relu.calculate_weighted_loss(x)
        x = self.relu(x)
        loss += self.lin2.calculate_weighted_loss(x)
        return loss

    def select_top_k_winners(self, k):
        wining_bit_width = {}
        for i, conv in enumerate(self.convs):
            mlp_winners = [*conv.nn.select_top_k_winners(k).values()]
            mp_winners = flatten_list([*conv.mp.select_top_k_winners(k).values()])
            wining_bit_width.update({f"conv_{i}": mlp_winners + [mp_winners]})
        wining_bit_width.update({f"lin1": flatten_list(self.lin1.select_top_k_winners(k).values())})
        wining_bit_width.update({f"relu": flatten_list(self.relu.select_top_k_winners(k).values())})
        wining_bit_width.update({f"lin2": flatten_list(self.lin2.select_top_k_winners(k).values())})
        return wining_bit_width

    def estimated_bit_operation_precision(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        bit_operations = 0
        for i, conv in enumerate(self.convs):
            bit_operations += conv.estimated_bit_operation_precision(x, edge_index)
            x = conv(x, edge_index)
        x = global_max_pool(x, batch)
        bit_operations += self.lin1.estimated_bit_operation_precision(x)
        x = self.lin1(x)
        bit_operations += self.relu.estimated_bit_operation_precision(x)
        x = self.relu(x)
        bit_operations += self.lin2.estimated_bit_operation_precision(x)
        return bit_operations


class QGIN(Module):
    def __init__(self,
                 in_channels,
                 num_layers,
                 hidden_channels,
                 out_channels,
                 num_bits,
                 ):
        super(QGIN, self).__init__()

        self.convs = ModuleList()
        for i in range(num_layers):
            is_first_layer = i == 0
            mlp_i = QLinearReLULinearReLU(in_channels if is_first_layer else hidden_channels,
                                          hidden_channels,
                                          hidden_channels,
                                          qi=False,
                                          qo=True,
                                          bit_widths=num_bits[i][0],
                                          quantize_per="column",
                                          )
            gin_i = QGraphIsomorphism(nn=mlp_i,
                                      qi=is_first_layer,
                                      qo=True,
                                      num_bits=num_bits[i][1],
                                      is_signed=True,
                                      quantize_per="column",
                                      )
            self.convs.append(gin_i)

        self.lin1 = QLinear(in_features=hidden_channels,
                            out_features=hidden_channels,
                            qi=True,
                            qo=True,
                            num_bits=num_bits[num_layers],
                            is_signed=False,
                            quantize_per="column",
                            )
        self.relu = QReLU(num_bits=num_bits[num_layers + 1], quantize_per="column", )
        self.lin2 = QLinear(in_features=hidden_channels,
                            out_features=out_channels,
                            qi=True,
                            qo=True,
                            num_bits=num_bits[num_layers + 2],
                            is_signed=False,
                            quantize_per="column",
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
            x = conv.simulated_quantize_forward(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.lin1.simulated_quantize_forward(x)
        x = self.relu.simulated_quantize_forward(x)
        x = self.lin2.simulated_quantize_forward(x)
        return x


def train(model, criterion, optimizer, loader, bit_width_lambda=None):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        classification_loss = criterion(out, data.y.view(-1))

        if bit_width_lambda is not None:
            bit_width_loss = bit_width_lambda * model.calculate_weighted_loss(data)
        else:
            bit_width_loss = tensor([0.0], device=device)

        (classification_loss + bit_width_loss).backward()
        total_loss += classification_loss.item() * data.num_graphs
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
def evaluate_loss(model, criterion, loader, device, bit_width_lambda=None):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss += criterion(out, data.y.view(-1)).item() * data.num_graphs
    classification_loss = loss / len(loader.dataset)
    if bit_width_lambda is not None:
        bit_width_loss = bit_width_lambda * model.calculate_weighted_loss(next(iter(loader)).to(device))
        return classification_loss + bit_width_loss.item()
    return classification_loss


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
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--step_size", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="logs")

    parser.add_argument("--num_bits_list", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--bit_width_lambda", type=float, default=1)

    args = parser.parse_args()
    arguments = vars(args)

    log_dir_name = (f"{args.dataset_name}/"
                    f"batch_size_{args.batch_size}/"
                    f"hidden_{args.hidden_channels}/"
                    f"layers_{args.num_layers}/"
                    f"wd_{format_fraction(args.weight_decay)}/"
                    f"lr_{format_fraction(args.lr)}/"
                    f"bit_width_{','.join(map(str, args.num_bits_list))}"
                    )
    if not exists(args.log_dir + "/" + log_dir_name):
        makedirs(args.log_dir + "/" + log_dir_name, exist_ok=True)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name = f"{log_dir_name}/log_{current_time}_bit_width_lambda_{format_fraction(args.bit_width_lambda)}"
    logger = setup_logger(filename=f"{args.log_dir}/{log_file_name}.log", verbose=True)

    [logger.info(f"{k}: {v}") for k, v in arguments.items()]
    logger.info("=" * 100)

    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root.joinpath("data")
    device = torch_device(args.device)
    dataset = TransformedTUDataset(root=data_dir, name=args.dataset_name, use_node_attr=True, use_edge_attr=True, cleaned=True)

    y = [d.y.item() for d in dataset]
    features_dim = dataset[0].x.shape[1]
    n_classes = len(np.unique(y))
    criterion = CrossEntropyLoss()
    splits = k_fold(dataset, args.num_folds)

    # logger.info(f"splits: {[[t.clone().tolist() for t in sublist] for sublist in splits]}")

    accuracies, bit_operations, wining_bit_widths_per_fold, average_bit_widths_per_fold = [], [], [], []
    for fold, (train_idx, val_idx, test_idx) in enumerate(zip(*splits)):
        model = MQGIN(in_channels=features_dim,
                      hidden_channels=args.hidden_channels,
                      out_channels=n_classes,
                      num_layers=args.num_layers,
                      num_bits_list=args.num_bits_list,
                      ).to(device)
        train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset, train_idx, val_idx, test_idx)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        best_state_dict, best_val_loss = None, float("inf")
        pbar = tqdm(range(args.epochs), desc="Epoch 0 Loss 0")
        for epoch in pbar:
            train_loss = train(model, criterion, optimizer, train_loader, args.bit_width_lambda)
            val_loss = evaluate_loss(model, criterion, val_loader, device, args.bit_width_lambda)
            scheduler.step()
            if val_loss < best_val_loss:
                best_state_dict = deepcopy(model.state_dict())
                best_val_loss = val_loss
                # print([*model.select_top_k_winners(1).values()])
            pbar.set_description(f"Epoch: {epoch}, "
                                 f"Train Loss: {train_loss:.2f}, "
                                 f"Val Loss: {val_loss:.2f}")
        model.load_state_dict(best_state_dict)
        wining_bit_widths = [*model.select_top_k_winners(1).values()]

        bit_ops = model.estimated_bit_operation_precision(next(iter(train_loader)).to(device))
        bit_operations += [bit_ops, ]
        wining_bit_widths_per_fold += [wining_bit_widths, ]
        average_bit_width = np.mean([*filter(lambda x: x != None, flatten_list(wining_bit_widths))])
        average_bit_widths_per_fold += [average_bit_width, ]

        logger.info(f"Average Bit-Width {average_bit_width:.2f}")
        logger.info(f"Winning bit widths: {wining_bit_widths}")

        model = QGIN(in_channels=features_dim,
                     hidden_channels=args.hidden_channels,
                     out_channels=n_classes,
                     num_layers=args.num_layers,
                     num_bits=[*model.select_top_k_winners(1).values()],
                     ).to(device)

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

        accuracies += [test_accuracies[best_epoch], ]

        logger.info(f"Fold {fold + 1}/{args.num_folds} - Accuracy: {test_accuracies[best_epoch]:.2f}%")

    wining_bit_widths = nested_median(*wining_bit_widths_per_fold)
    wining_bit_widths_std = nested_std(*wining_bit_widths_per_fold)

    logger.info("=" * 100)

    logger.info(f"Results for {args.num_folds}-fold cross-validation:")
    logger.info(f"winning bit widths: {wining_bit_widths}")
    logger.info(f"winning bit widths std: {wining_bit_widths_std}")
    logger.info(f"average bit width: {np.mean(average_bit_widths_per_fold):.2f} ± {np.std(average_bit_widths_per_fold):.2f}")
    logger.info(f"Accuracies: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    logger.info(f"Bit Operations: {np.median(bit_operations):.2f} ± {np.std(bit_operations):.2f}")

    write_to_csv(f"{args.log_dir}/{log_dir_name}/results.csv", {"bit_width_lambda": args.bit_width_lambda,
                                                                "accuracy_mean": np.mean(accuracies),
                                                                "accuracy_std": np.std(accuracies),
                                                                "bit_operations_mean": np.median(bit_operations),
                                                                "bit_operations_std": np.std(bit_operations),
                                                                "average_bit_width_mean": np.mean(average_bit_widths_per_fold),
                                                                "average_bit_width_std": np.std(average_bit_widths_per_fold),
                                                                # "wining_bit_widths": wining_bit_widths,
                                                                # "wining_bit_widths_std": wining_bit_widths_std
                                                                })
