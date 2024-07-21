from datetime import datetime
from os import makedirs
from os.path import exists

import numpy as np
import argparse
from collections import defaultdict

from prettytable import PrettyTable
from torch import cuda, device as torch_device
from torch.nn import Sequential, BatchNorm1d, Module, Linear
from torch_geometric.nn import global_mean_pool, GCNConv

from tasks_per_graph.synthetic.CSL import CSL
from tasks_per_graph.synthetic.synthetic_utility import positional_encoding, classification_task
from tasks_per_graph.synthetic.loader import prepare_loaders
from training.logger import setup_logger


class GCN(Module):
    def __init__(self, dim_features, dim_embedding, dim_target, num_layers):
        super(GCN, self).__init__()

        self.embedding_pos_enc = Linear(in_features=dim_features,
                                        out_features=dim_embedding,
                                        )
        self.convs = Sequential(*[GCNConv(in_channels=dim_embedding,
                                          out_channels=dim_embedding,
                                          )
                                  for i in range(num_layers)])
        self.bns = Sequential(*[BatchNorm1d(dim_embedding) for _ in range(num_layers)])
        self.fc1 = Linear(in_features=dim_embedding,
                          out_features=dim_embedding // 2,
                          )
        self.fc2 = Linear(in_features=dim_embedding // 2,
                           out_features=dim_embedding // 4,
                           )
        self.fc3 = Linear(in_features=dim_embedding // 4,
                           out_features=dim_target,
                           )

    def reset_parameters(self):
        self.embedding_pos_enc.reset_parameters()
        for layer in self.convs:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding_pos_enc(x)
        for conv, bn in zip(self.convs, self.bns):
            x_in = x
            x = conv(x, edge_index)
            x = bn(x)
            x = x.relu()
            x = x_in + x
        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
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

        model = GCN(**model_config, dim_features=node_features).to(device)

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
    args = parser.parse_args()
    arguments = vars(args)

    log_dir_name = f"CSL_{args.num_layers}-layers_fp32"
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

    train_config = dict(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

    results = defaultdict(list)
    for run in range(args.runs):
        logger.info(f"Run {run + 1}/{args.runs}")
        run_i_results = main_classification(dataset, model_config, train_config, device)
        results["train"] += run_i_results["train"].tolist()
        results["val"] += run_i_results["val"].tolist()
        results["test"] += run_i_results["test"].tolist()

    table = PrettyTable()
    table.title = f"CSL Results with {args.num_layers} layers and FP32 bit width"
    table.field_names = ["Train Mean ± Std", "Train Min", "Train Max", "Test Mean ± Std", "Test Min", "Test Max"]
    table.add_row([f"{np.mean(results['train']):.2f} ± {np.std(results['train']):.2f}",
                   f"{np.min(results['train']):.2f}",
                   f"{np.max(results['train']):.2f}",
                   f"{np.mean(results['test']):.2f} ± {np.std(results['test']):.2f}",
                   f"{np.min(results['test']):.2f}",
                   f"{np.max(results['test']):.2f}"])
    logger.info(table)
