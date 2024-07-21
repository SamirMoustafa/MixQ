from datetime import datetime
from os import makedirs
from os.path import exists
from typing import List

import numpy as np
import argparse
from collections import defaultdict

from prettytable import PrettyTable
from torch import cuda, device as torch_device
from torch.nn import Sequential, BatchNorm1d, Module
from torch_geometric.nn import global_mean_pool

from quantization.fixed_modules.non_parametric import QReLU
from quantization.fixed_modules.parametric import QGraphConvolution, QLinear
from tasks_per_graph.synthetic.CSL import CSL

from tasks_per_graph.synthetic.synthetic_utility import positional_encoding, classification_task
from tasks_per_graph.synthetic.loader import prepare_loaders
from training.logger import setup_logger


class QGCN(Module):
    def __init__(self, dim_features, dim_embedding, dim_target, num_layers, num_bits: List[int]):
        super(QGCN, self).__init__()

        self.embedding_pos_enc = QLinear(in_features=dim_features,
                                         out_features=dim_embedding,
                                         qi=True,
                                         qo=True,
                                         num_bits=num_bits[0],
                                         is_signed=False,
                                         quantize_per="column",
                                         )
        self.convs = Sequential(*[QGraphConvolution(in_channels=dim_embedding,
                                                    out_channels=dim_embedding,
                                                    # All layer's input should be quantized due to the residual
                                                    # Expect the first layer, which is quantized due to the embedding
                                                    qi=i != 0,
                                                    qo=True,
                                                    num_bits=num_bits[i + 1],
                                                    is_signed=False,
                                                    quantize_per="column",
                                                    cached=False,
                                                    )
                                  for i in range(num_layers)])
        self.relus = Sequential(*[QReLU(qi=False,
                                        num_bits=num_bits[i + num_layers + 1],
                                        )
                                  for i in range(num_layers)])
        self.bns = Sequential(*[BatchNorm1d(dim_embedding) for _ in range(num_layers)])
        self.fc1 = QLinear(in_features=dim_embedding,
                           out_features=dim_embedding // 2,
                           # Input should be quantized due to the mean pooling
                           qi=True,
                           qo=True,
                           num_bits=num_bits[2 * num_layers + 1],
                           is_signed=False,
                           quantize_per="column",
                           )
        self.relu1 = QReLU(qi=False,
                           num_bits=num_bits[2 * num_layers + 2],
                           )
        self.fc2 = QLinear(in_features=dim_embedding // 2,
                           out_features=dim_embedding // 4,
                           qi=False,
                           qo=True,
                           num_bits=num_bits[2 * num_layers + 3],
                           is_signed=False,
                           quantize_per="column",
                           )
        self.relu2 = QReLU(qi=False,
                           num_bits=num_bits[2 * num_layers + 4],
                           )
        self.fc3 = QLinear(in_features=dim_embedding // 4,
                           out_features=dim_target,
                           qi=False,
                           qo=True,
                           num_bits=num_bits[2 * num_layers + 5],
                           is_signed=False,
                           quantize_per="column",
                           )

    def reset_parameters(self):
        self.embedding_pos_enc.reset_parameters()
        for layer in self.convs:
            layer.reset_parameters()
        for layer in self.relus:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()
        self.fc1.reset_parameters()
        self.relu1.reset_parameters()
        self.fc2.reset_parameters()
        self.relu2.reset_parameters()
        self.fc3.reset_parameters()

    def set_forward_func(self, forward_func: callable):
        self.forward = forward_func

    def full_precision_forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding_pos_enc(x)
        for conv, bn, relu in zip(self.convs, self.bns, self.relus):
            x_in = x
            x = conv(x, edge_index)
            x = bn(x)
            x = relu(x)
            x = x_in + x
        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def simulated_quantize_forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding_pos_enc.simulated_quantize_forward(x)
        for conv, bn, relu in zip(self.convs, self.bns, self.relus):
            x_in = x
            x = conv.simulated_quantize_forward(x, edge_index)
            x = bn(x)
            x = relu.simulated_quantize_forward(x)
            x = x_in + x
        x = global_mean_pool(x, data.batch)
        x = self.fc1.simulated_quantize_forward(x)
        x = self.relu1.simulated_quantize_forward(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def main_classification(dataset, model_config, train_config, device):
    splits = dataset.get_all_splits_idx()
    results = dict(train=[], val=[], test=[])
    n_splits = dataset.n_splits
    node_features = dataset[0].x.shape[1]

    for index in range(n_splits):
        loaders = prepare_loaders(dataset, splits, batch_size=train_config["batch_size"], index=index)
        logger.info(f"Split {index + 1}/{n_splits}")

        model = QGCN(**model_config, dim_features=node_features).to(device)

        model.set_forward_func(model.full_precision_forward)
        train_acc, val_acc, test_acc = classification_task(model,
                                                           loaders,
                                                           epochs=train_config["epochs"],
                                                           lr=train_config["lr"],
                                                           device=device)
        model.set_forward_func(model.simulated_quantize_forward)
        train_acc, val_acc, test_acc = classification_task(model,
                                                           loaders,
                                                           epochs=train_config["epochs"],
                                                           lr=train_config["lr"],
                                                           device=device)
        logger.info(f"Train Acc {train_acc:.2f} Val Acc {val_acc:.2f} Test Acc {test_acc:.2f}")
        results["train"].append(train_acc)
        results["val"].append(val_acc)
        results["test"].append(test_acc)

    return {k: np.asarray(v) for k, v in results.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN baselines on synthetic datasets")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs for each experiment")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs (default: 500)")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers (default: 4)")
    parser.add_argument("--dim_embedding", type=int, default=150, help="Dimension of node embedding (default: 150)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Choose between cuda or cpu")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save the logs")
    parser.add_argument("--bit_width", type=int, default=4, help="Bit width for all layers (default: 4)")
    args = parser.parse_args()
    arguments = vars(args)

    log_dir_name = f"CSL_{args.num_layers}-layers_{args.bit_width}-bitwidth"
    if not exists(args.log_dir + "/" + log_dir_name):
        makedirs(args.log_dir + "/" + log_dir_name, exist_ok=True)

    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name = f"{log_dir_name}/log_{current_time}"
    logger = setup_logger(filename=f"{args.log_dir}/{log_file_name}.log", verbose=True)
    [logger.info(f"{k}: {v}") for k, v in arguments.items()]
    logger.info("=" * 100)

    device = torch_device(args.device)
    print(f"PROCESSING ...")
    dataset = CSL()
    for data_i in dataset:
        data_i.x = positional_encoding(data_i.edge_index)

    model_config = dict(dim_embedding=args.dim_embedding, dim_target=dataset.num_classes, num_layers=args.num_layers)

    # Set the bit width for each layer
    model_config["num_bits"] = [[args.bit_width, args.bit_width, args.bit_width]]  # First positional encoding layer
    model_config["num_bits"] += [[args.bit_width, args.bit_width, args.bit_width, args.bit_width, args.bit_width, args.bit_width]
                                 # All GCN layers have 6 components
                                 for _ in range(args.num_layers)]
    model_config["num_bits"] += [[args.bit_width] for _ in range(args.num_layers)]  # All ReLU layers have 1 component
    model_config["num_bits"] += [[args.bit_width, args.bit_width, args.bit_width],  # First FC layer
                                 [args.bit_width],  # First ReLU layer
                                 [args.bit_width, args.bit_width, args.bit_width],  # Second FC layer
                                 [args.bit_width],  # Second ReLU layer
                                 [args.bit_width, args.bit_width, args.bit_width],  # Third FC layer
                                 ]

    train_config = dict(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

    results = defaultdict(list)
    for run in range(args.runs):
        logger.info(f"Run {run + 1}/{args.runs}")
        run_i_results = main_classification(dataset, model_config, train_config, device)
        results["train"] += run_i_results["train"].tolist()
        results["val"] += run_i_results["val"].tolist()
        results["test"] += run_i_results["test"].tolist()

    table = PrettyTable()
    table.title = f"CSL Results with {args.num_layers} layers and {args.bit_width} bit width"
    table.field_names = ["Train Mean ± Std", "Train Min", "Train Max", "Test Mean ± Std", "Test Min", "Test Max"]
    table.add_row([f"{np.mean(results['train']):.2f} ± {np.std(results['train']):.2f}",
                   f"{np.min(results['train']):.2f}",
                   f"{np.max(results['train']):.2f}",
                   f"{np.mean(results['test']):.2f} ± {np.std(results['test']):.2f}",
                   f"{np.min(results['test']):.2f}",
                   f"{np.max(results['test']):.2f}"])
    logger.info(table)
