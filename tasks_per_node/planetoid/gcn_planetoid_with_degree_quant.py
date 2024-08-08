import argparse
from datetime import datetime
from os import makedirs
from os.path import exists
from pathlib import Path
from typing import List

import numpy as np
from numpy import std, mean
from torch_operation_counter import OperationsCounterMode
from tqdm import tqdm

from torch.nn import Module, ModuleList
from torch.optim import Adam
from torch import cuda, bernoulli, device, no_grad, tensor, bool as torch_bool
from torch.nn.functional import cross_entropy, dropout

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, Compose

from quantization.fixed_modules.parametric import MaskQuantGraphConvolution
from training.logger import setup_logger
from training.probability_degree_transforms import ProbabilisticHighDegreeMask
from training.tensorboard_logger import TensorboardLogger
from quantization.fixed_modules.non_parametric import QReLU
from quantization.mixed_modules.non_parametric.activations import MQReLU
from quantization.mixed_modules.parametric.graph_convolution import MQGraphConvolution
from utility import format_fraction, nested_median, nested_std, write_to_csv, flatten_list


class MQGCN(Module):
    def __init__(self, num_channels: int, hidden_channels: int, out_channels: int, num_bits: List[int], number_of_layers: int):
        super(MQGCN, self).__init__()
        self.number_of_layers = number_of_layers
        self.convs = ModuleList()
        self.relus = ModuleList()
        for i in range(number_of_layers):
            is_first_layer = i == 0
            is_last_layer = i == number_of_layers - 1
            gcn_i = MQGraphConvolution(in_channels=hidden_channels if not is_first_layer else num_channels,
                                       out_channels=hidden_channels if not is_last_layer else out_channels,
                                       qi=is_first_layer,
                                       qo=True,
                                       num_bits_list=num_bits,
                                       is_signed=False,
                                       quantize_per="column",
                                       )
            self.convs.append(gcn_i)
            if not is_last_layer:
                relu_i = MQReLU(num_bits_list=num_bits, quantize_per="column")
                self.relus.append(relu_i)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for relu in self.relus:
            relu.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.number_of_layers):
            is_last_layer = i == self.number_of_layers - 1
            x = self.convs[i](x, edge_index, edge_attr)
            if not is_last_layer:
                x = self.relus[i](x)
        return x

    def calculate_loss(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        loss = 0
        for i in range(self.number_of_layers):
            is_last_layer = i == self.number_of_layers - 1
            loss += self.convs[i].calculate_weighted_loss(x, edge_index, edge_attr)
            x = self.convs[i](x, edge_index, edge_attr)
            if not is_last_layer:
                loss += self.relus[i].calculate_weighted_loss(x)
                x = self.relus[i](x)
        return loss

    @no_grad()
    def estimated_bit_operation_precision(self, data):
        x, edge_index = data.x, data.edge_index
        bit_operations = 0
        for i in range(self.number_of_layers):
            is_last_layer = i == self.number_of_layers - 1
            bit_operations += self.convs[i].estimated_bit_operation_precision(x, edge_index)
            x = self.convs[i](x, edge_index)
            if not is_last_layer:
                bit_operations += self.relus[i].estimated_bit_operation_precision(x)
                x = self.relus[i](x)
        return bit_operations

    def get_best_bit_config(self):
        bit_config = []
        for i in range(self.number_of_layers):
            is_last_layer = i == self.number_of_layers - 1
            # bit_config += [[sum(self.convs[i].select_top_k_winners(1).values(), []),
            #                 sum(self.relus[i].select_top_k_winners(1).values(), []),], ]
            conv_i_bit_config = sum(self.convs[i].select_top_k_winners(1).values(), [])
            if not is_last_layer:
                relu_i_bit_config = sum(self.relus[i].select_top_k_winners(1).values(), [])
            else:
                relu_i_bit_config = []
            bit_config += [[conv_i_bit_config, relu_i_bit_config], ]
        return bit_config


class QGCN(Module):
    def __init__(self, num_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_bits: List[int],
                 number_of_layers: int,
                 percentile,
                 use_momentum,
                 momentum,
                 ):
        super(QGCN, self).__init__()
        self.number_of_layers = number_of_layers

        self.convs = ModuleList()
        self.relus = ModuleList()
        for i in range(number_of_layers):
            is_first_layer = i == 0
            is_last_layer = i == number_of_layers - 1
            gcn_i = MaskQuantGraphConvolution(in_channels=hidden_channels if not is_first_layer else num_channels,
                                              out_channels=hidden_channels if not is_last_layer else out_channels,
                                              qi=is_first_layer,
                                              qo=True,
                                              num_bits=num_bits[i][0],
                                              is_signed=True,
                                              quantize_per="column",
                                              quant_use_momentum=use_momentum,
                                              quant_momentum=momentum,
                                              quant_percentile=percentile,
                                              )
            self.convs.append(gcn_i)
            if not is_last_layer:
                relu_i = QReLU(num_bits=num_bits[i][1], quantize_per="column")
                self.relus.append(relu_i)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for relu in self.relus:
            relu.reset_parameters()

    def set_forward_func(self, forward_func: callable):
        self.forward = forward_func

    def full_precision_forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_in = None
        for i in range(self.number_of_layers):
            is_first_layer = i == 0
            is_last_layer = i == self.number_of_layers - 1
            x = self.convs[i](x, edge_index, edge_attr)
            if is_first_layer:
                x_in = x
            if not is_last_layer:
                x = self.relus[i](x)
                x = dropout(x, p=0.5, training=self.training)
                x = x_in + x
        return x

    def simulated_quantize_forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        binary_mask = bernoulli(data.probability_mask).to(torch_bool)
        x_in = None
        for i in range(self.number_of_layers):
            is_first_layer = i == 0
            is_last_layer = i == self.number_of_layers - 1
            x = self.convs[i].simulated_quantize_forward(x, edge_index, binary_mask=binary_mask)
            if is_first_layer:
                x_in = x
            if not is_last_layer:
                x = self.relus[i].simulated_quantize_forward(x)
                x = dropout(x, p=0.5, training=self.training)
                x = x_in + x
        return x


def train(model, optimizer, data, bit_width_lambda=None):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    classification_loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
    if bit_width_lambda is not None:
        bit_width_loss = bit_width_lambda * model.calculate_loss(data)
    else:
        bit_width_loss = tensor([0.0], device=out.device)
    (classification_loss + bit_width_loss).backward()
    optimizer.step()
    return classification_loss.item()


@no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data).argmax(dim=-1)
    accuracies = [(pred[mask] == data.y[mask]).float().mean().item() * 100
                  for mask in (data.train_mask, data.val_mask, data.test_mask)]
    return accuracies


def training_loop(epochs, model, optimizer, data, bit_width_lambda=None, log_directory=None):
    directory = Path(log_directory) if log_directory is not None else None
    directory.mkdir(parents=True, exist_ok=True) if log_directory is not None else None
    if log_directory is not None:
        tensorboard_logger = TensorboardLogger(log_directory)
    best_val_accuracy, best_epoch, test_accuracy = -float("inf"), 0, 0
    best_state_dict = model.state_dict()
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        loss = train(model, optimizer, data, bit_width_lambda)
        train_accuracy, validation_accuracy, tmp_test_accuracy = evaluate(model, data)
        if validation_accuracy > best_val_accuracy:
            best_epoch = epoch
            best_val_accuracy = validation_accuracy
            test_accuracy = tmp_test_accuracy
            best_state_dict = model.state_dict().copy()
        pbar.set_description(f"{epoch:03d}/{epochs:03d},Loss:{loss:.2f},TrainAcc:{train_accuracy:.2f},"
                             f"ValAcc:{validation_accuracy:.2f},BestValAcc:{best_val_accuracy:.2f},BestEpoch:{best_epoch:03d}")
        training_dict = {"loss": loss,
                         "learning rate": optimizer.param_groups[0]["lr"],
                         }
        training_dict.update({"accuracy": train_accuracy})
        validation_dict = {"accuracy": validation_accuracy}
        if log_directory is not None:
            tensorboard_logger.training_epoch_end(model, epoch, training_dict, validation_dict)
    return best_state_dict, test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="PubMed")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--number_of_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_quant", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="experimental_plus_DQ_logs")

    parser.add_argument("--num_bits_list", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--bit_width_lambda", type=float, default=-1e-8)
    parser.add_argument("--mask_low_probability", type=float, default=0.0)
    parser.add_argument("--mask_high_probability", type=float, default=0.1)
    parser.add_argument("--quant_percentile", type=float, default=0.01)
    parser.add_argument("--quant_use_momentum", type=bool, default=True)
    parser.add_argument("--quant_momentum", type=float, default=0.01)
    args = parser.parse_args()

    arguments = vars(args)

    log_dir_name = (f"{args.dataset_name}/"
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

    device = device(args.device)

    high_probability = min(args.mask_low_probability + args.mask_high_probability, 1.0)
    mask_transform = ProbabilisticHighDegreeMask(args.mask_low_probability, high_probability)

    dataset = Planetoid(root="../../data", name=args.dataset_name, transform=Compose([NormalizeFeatures(),
                                                                                      mask_transform]))
    data = dataset[0].to(device)

    full_precision_accuracies, quantized_accuracies, wining_bit_widths, average_bit_widths, quantized_bit_operations = [], [], [], [], []
    for run_i in range(args.num_runs):
        model = MQGCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args.num_bits_list, args.number_of_layers)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_state_dict, test_accuracy = training_loop(args.epochs, model, optimizer, data, args.bit_width_lambda)
        model.load_state_dict(best_state_dict)

        logger.info(f"[{run_i + 1}/{args.num_runs}]: Relaxed Model, Accuracy: {test_accuracy:.2f}%")

        best_bit_config = model.get_best_bit_config()
        bit_operation = model.estimated_bit_operation_precision(data)
        average_bit_width = mean([*filter(lambda x: x != None, flatten_list(best_bit_config))])

        average_bit_widths += [average_bit_width]
        wining_bit_widths += [best_bit_config]
        quantized_bit_operations += [bit_operation]

        logger.info(f"Expected bit operations: {bit_operation}")
        logger.info(f"Average bit width: {average_bit_width}")

        logger.info(f"Winning Bit Width: {[*filter(lambda x: x != None, flatten_list(best_bit_config))]}")

        model = QGCN(num_channels=dataset.num_features,
                     hidden_channels=args.hidden_channels,
                     out_channels=dataset.num_classes,
                     num_bits=best_bit_config,
                     number_of_layers=args.number_of_layers,
                     percentile=args.quant_percentile,
                     use_momentum=args.quant_use_momentum,
                     momentum=args.quant_momentum,
                     ).to(device)

        model.set_forward_func(model.full_precision_forward)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_state_dict, test_accuracy = training_loop(args.epochs, model, optimizer, data)
        logger.info(f"[{run_i + 1}/{args.num_runs}]: Full precision Accuracy: {test_accuracy:.2f}%")
        full_precision_accuracies.append(test_accuracy)
        with no_grad():
            with OperationsCounterMode(model) as ops_counter:
                model.full_precision_forward(data)
        full_precision_bit_operations = ops_counter.total_main_operation * 32

        model.load_state_dict(best_state_dict)
        model.set_forward_func(model.simulated_quantize_forward)
        optimizer = Adam(model.parameters(), lr=args.lr_quant)
        best_state_dict, test_accuracy = training_loop(args.epochs, model, optimizer, data)
        logger.info(f"[{run_i + 1}/{args.num_runs}]: Quantized Accuracy: {test_accuracy:.2f}%")
        quantized_accuracies.append(test_accuracy)
        logger.info("=" * 100)

    logger.info(f"Full precision BOPs: {full_precision_bit_operations:.2f}")
    logger.info(f"Full precision accuracy: {mean(full_precision_accuracies):.2f} ± {std(full_precision_accuracies):.2f}")

    logger.info(f"Average wining bit configuration: {nested_median(*wining_bit_widths)}")
    logger.info(f"Standard deviation of wining bit configuration: {nested_std(*wining_bit_widths)}")

    logger.info(f"Bit width: {mean(average_bit_widths):.2f} ± {std(average_bit_widths):.2f}")
    logger.info(f"Quantized BOPs: {mean(quantized_bit_operations):.2f} ± {std(quantized_bit_operations):.2f}")
    logger.info(f"Quantized accuracy: {mean(quantized_accuracies):.2f} ± {std(quantized_accuracies):.2f}")
    write_to_csv(f"{args.log_dir}/{log_dir_name}/results.csv", {"bit_width_lambda": args.bit_width_lambda,
                                                                "accuracy_mean": np.mean(quantized_accuracies),
                                                                "accuracy_std": np.std(quantized_accuracies),
                                                                "bit_operations_mean": np.median(quantized_bit_operations),
                                                                "bit_operations_std": np.std(quantized_bit_operations),
                                                                "average_bit_width_mean": np.mean(average_bit_widths),
                                                                "average_bit_width_std": np.std(average_bit_widths),
                                                                })

