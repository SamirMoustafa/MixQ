# Maximize GPU usage is 12032 MiB
import copy
import argparse
from datetime import datetime
from os import makedirs
from os.path import exists
from pathlib import Path

from numpy import mean, std, median
from tqdm import tqdm

from torch.nn import Module
from torch.optim import NAdam
from torch import no_grad, cat, tensor
from torch import cuda, device as torch_device
from torch.nn.functional import dropout, cross_entropy
from torchmetrics import Accuracy
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader


from rebuttal_phase.SAGEConv import QGraphSAGE, MQGraphSAGE
from quantization.fixed_modules.non_parametric import QReLU
from quantization.mixed_modules.non_parametric import MQReLU
from training.logger import setup_logger
from utility import format_fraction, nested_median, nested_std, write_to_csv


class MQSAGE(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_bits):
        super().__init__()
        self.sage_1 = MQGraphSAGE(in_channels=in_channels,
                                  out_channels=hidden_channels,
                                  qi=True,
                                  qo=True,
                                  num_bits_list=num_bits,
                                  is_signed=False,
                                  quantize_per="column",
                                  )
        self.relu_1 = MQReLU(num_bits_list=num_bits, quantize_per="column")
        self.sage_2 = MQGraphSAGE(in_channels=hidden_channels,
                                  out_channels=out_channels,
                                  qi=False,
                                  qo=True,
                                  num_bits_list=num_bits,
                                  quantize_per="column",
                                  )
        self.reset_parameters()

    def reset_parameters(self):
        self.sage_1.reset_parameters()
        self.relu_1.reset_parameters()
        self.sage_2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.sage_1(x, edge_index)
        x = self.relu_1(x)
        x = self.sage_2(x, edge_index)
        return x

    def calculate_loss(self, x, edge_index):
        loss = self.sage_1.calculate_weighted_loss(x, edge_index)
        x = self.sage_1(x, edge_index)
        loss += self.relu_1.calculate_weighted_loss(x)
        x = self.relu_1(x)
        loss += self.sage_2.calculate_weighted_loss(x, edge_index)
        return loss

    @no_grad()
    def inference(self, x_all, subgraph_loader):
        device = x_all.device
        for i, conv in enumerate([self.sage_1, self.sage_2]):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                x = conv(x.to(device), batch.edge_index.to(device))
                if i < 1:
                    x = self.relu_1(x)
                xs.append(x[:batch.batch_size].detach().cpu())
            x_all = cat(xs, dim=0)
        return x_all



class QSAGE(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_bits):
        super().__init__()
        self.sage_1 = QGraphSAGE(in_channels=in_channels,
                                 out_channels=hidden_channels,
                                 qi=True,
                                 qo=True,
                                 num_bits=num_bits[0],
                                 is_signed=False,
                                 quantize_per="column",
                                 )
        self.relu_1 = QReLU(num_bits=num_bits[1], quantize_per="column")
        self.sage_2 = QGraphSAGE(in_channels=hidden_channels,
                                 out_channels=out_channels,
                                 qi=False,
                                 qo=True,
                                 num_bits=num_bits[2],
                                 quantize_per="column",
                                 )
        self.reset_parameters()

    def reset_parameters(self):
        self.sage_1.reset_parameters()
        self.relu_1.reset_parameters()
        self.sage_2.reset_parameters()

    def set_forward_func(self, forward_func):
        self.forward = forward_func

    def full_precision_forward(self, x, edge_index):
        x = self.sage_1(x, edge_index)
        x = self.relu_1(x)
        x = dropout(x, p=0.5, training=self.training)
        x = self.sage_2(x, edge_index)
        return x

    def simulated_quantize_forward(self, x, edge_index):
        x = self.sage_1.simulated_quantize_forward(x, edge_index)
        x = self.relu_1.simulated_quantize_forward(x)
        x = dropout(x, p=0.5, training=self.training)
        x = self.sage_2.simulated_quantize_forward(x, edge_index)
        return x

    @no_grad()
    def inference(self, x_all, subgraph_loader):
        device = x_all.device
        for i, conv in enumerate([self.sage_1, self.sage_2]):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                x = conv(x.to(device), batch.edge_index.to(device))
                if i < 1:
                    x = self.relu_1(x)
                xs.append(x[:batch.batch_size].detach().cpu())
            x_all = cat(xs, dim=0)
        return x_all

    def estimated_bit_operation_precision(self, x_all, subgraph_loader):
        bit_operation_per_layer = []
        device = x_all.device
        for i, conv in enumerate([self.sage_1, self.sage_2]):
            xs = []
            bit_operation_per_batch = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                bit_operation = conv.estimated_bit_operation_precision(x.to(device), batch.edge_index.to(device))
                bit_operation_per_batch += [bit_operation]
                x = conv(x.to(device), batch.edge_index.to(device))
                if i < 1:
                    bit_operation_per_batch += [self.relu_1.estimated_bit_operation_precision(x)]
                    x = self.relu_1(x)
                xs.append(x[:batch.batch_size].detach().cpu())
            bit_operation_per_layer += [mean(bit_operation_per_batch)]
            x_all = cat(xs, dim=0)
        return sum(bit_operation_per_layer)


def train(epoch, model, train_loader, optimizer, bit_width_lambda=None):
    model.train()
    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        classification_loss = cross_entropy(y_hat, y)
        if bit_width_lambda is not None:
            bit_width_loss = bit_width_lambda * model.calculate_loss(batch.x, batch.edge_index.to(device))
        else:
            bit_width_loss = tensor([0.0], device=y.device)
        (classification_loss + bit_width_loss).backward()
        optimizer.step()

        total_loss += float(classification_loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
    loss = total_loss / total_examples
    return loss


@no_grad()
def evaluate(model, data, subgraph_loader):
    model.eval()
    accuracies = []
    accuracy = Accuracy(num_classes=data.y.max().item() + 1, task="multiclass")
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    for mask in (data.train_mask, data.val_mask, data.test_mask):
        accuracies += [accuracy(y_hat[mask], data.y[mask].cpu()).item() * 100]
    return accuracies


def training_loop(epochs, model, optimizer, data, train_loader, subgraph_loader, bit_width_lambda=None, log_directory=None):
    directory = Path(log_directory) if log_directory is not None else None
    directory.mkdir(parents=True, exist_ok=True) if log_directory is not None else None
    best_val_accuracy, best_epoch, test_accuracy = -float("inf"), 0, 0
    best_state_dict = model.state_dict()
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        loss = train(epoch, model, train_loader, optimizer, bit_width_lambda)
        train_accuracy, validation_accuracy, tmp_test_accuracy = evaluate(model, data, subgraph_loader)
        if validation_accuracy > best_val_accuracy:
            best_epoch = epoch
            best_val_accuracy = validation_accuracy
            test_accuracy = tmp_test_accuracy
            best_state_dict = model.state_dict().copy()
        pbar.set_description(f"{epoch:03d}/{epochs:03d},"
                             f"Loss:{loss:.2f},"
                             f"TrainAcc:{train_accuracy:.2f},"
                             f"ValAcc:{validation_accuracy:.2f},"
                             f"BestValAcc:{best_val_accuracy:.2f},"
                             f"BestEpoch:{best_epoch:03d}")
    return best_state_dict, test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--lr_quant", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="experimental_logs")

    parser.add_argument("--num_bits_list", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--bit_width_lambda", type=float, default=1)

    args = parser.parse_args()
    arguments = vars(args)

    log_dir_name = (f"dataset_Reddit/"
                    f"GraphSAGE/"
                    f"hidden_{args.hidden_channels}/"
                    f"wd_{format_fraction(args.weight_decay)}/"
                    f"lr_{format_fraction(args.lr)}/"
                    f"bit_width_{','.join(map(str, args.num_bits_list))}"
                    )
    if not exists(args.log_dir + "/" + log_dir_name):
        makedirs(args.log_dir + "/" + log_dir_name)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name = f"log_bit_width_lambda_{format_fraction(args.bit_width_lambda)}_{current_time}"
    logger = setup_logger(filename=f"{args.log_dir}/{log_dir_name}/{log_file_name}.log", verbose=True)

    [logger.info(f"{k}: {v}") for k, v in arguments.items()]
    logger.info("=" * 100)

    device = torch_device(args.device)
    dataset = Reddit("../data/reddit/")
    num_nodes = dataset[0].num_nodes
    num_edges = dataset[0].num_edges

    data = dataset[0].to(device, "x", "y")
    kwargs = {"batch_size": args.batch_size, "num_workers": 6, "persistent_workers": True}
    train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[25, 10], shuffle=True, **kwargs)
    subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None, num_neighbors=[-1], shuffle=False, **kwargs)

    full_precision_accuracies, quantized_accuracies, wining_bit_widths, average_bit_widths, quantized_bit_operations = [], [], [], [], []
    for run_i in range(args.num_runs):
        model = MQSAGE(dataset.num_features, args.hidden_channels, dataset.num_classes, args.num_bits_list)
        model = model.to(device)

        optimizer = NAdam(model.parameters(), lr=args.lr_quant, weight_decay=args.weight_decay)

        best_state_dict, test_accuracy = training_loop(args.epochs, model, optimizer, data, train_loader, subgraph_loader, args.bit_width_lambda, args.log_dir + "/" + log_dir_name)
        logger.info(f"[{run_i + 1}/{args.num_runs}]: Relaxed Model, Accuracy: {test_accuracy:.2f}%")

        model.load_state_dict(best_state_dict)

        wining_bit_width = [sum(model.sage_1.select_top_k_winners(1).values(), []),
                            sum(model.relu_1.select_top_k_winners(1).values(), []),
                            sum(model.sage_2.select_top_k_winners(1).values(), []),
                            ]
        average_bit_width = mean([*filter(lambda x: x != None, sum(wining_bit_width, []))])
        average_bit_widths += [average_bit_width]
        wining_bit_widths += [wining_bit_width]

        logger.info("=" * 100)
        logger.info(f"Winning Bit Width: {[*filter(lambda x: x != None, sum(wining_bit_width, []))]}")

        model = QSAGE(dataset.num_features, args.hidden_channels, dataset.num_classes, wining_bit_width).to(device)

        bit_operation = model.estimated_bit_operation_precision(data.x, subgraph_loader)
        logger.info(f"Expected bit operations: {bit_operation}")
        quantized_bit_operations.append(bit_operation)

        model.set_forward_func(model.full_precision_forward)
        optimizer = NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_state_dict, test_accuracy = training_loop(args.epochs, model, optimizer, data, train_loader, subgraph_loader, None, args.log_dir + "/" + log_dir_name)

        logger.info(f"[{run_i + 1}/{args.num_runs}]: Full precision Accuracy: {test_accuracy:.2f}%")
        full_precision_accuracies.append(test_accuracy)

        model.load_state_dict(best_state_dict)
        model.set_forward_func(model.simulated_quantize_forward)
        optimizer = NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_state_dict, test_accuracy = training_loop(args.epochs, model, optimizer, data, train_loader, subgraph_loader, None, args.log_dir + "/" + log_dir_name)
        logger.info(f"[{run_i + 1}/{args.num_runs}]: Quantized Accuracy: {test_accuracy:.2f}%")
        quantized_accuracies.append(test_accuracy)

    logger.info("=" * 80)
    logger.info(f"Full precision accuracy: {mean(full_precision_accuracies):.2f} ± {std(full_precision_accuracies):.2f}")

    logger.info(f"Average wining bit configuration: {nested_median(*wining_bit_widths)}")
    logger.info(f"Standard deviation of wining bit configuration: {nested_std(*wining_bit_widths)}")

    logger.info(f"Bit width: {mean(average_bit_widths):.2f} ± {std(average_bit_widths):.2f}")
    logger.info(f"Quantized BOPs: {mean(quantized_bit_operations):.2f} ± {std(quantized_bit_operations):.2f}")
    logger.info(f"Quantized accuracy: {mean(quantized_accuracies):.2f} ± {std(quantized_accuracies):.2f}")
    write_to_csv(f"{args.log_dir}/{log_dir_name}/results.csv", {"bit_width_lambda": args.bit_width_lambda,
                                                                "accuracy_mean": mean(quantized_accuracies),
                                                                "accuracy_std": std(quantized_accuracies),
                                                                "bit_operations_mean": median(quantized_bit_operations),
                                                                "bit_operations_std": std(quantized_bit_operations),
                                                                "average_bit_width_mean": mean(average_bit_widths),
                                                                "average_bit_width_std": std(average_bit_widths),
                                                                })
