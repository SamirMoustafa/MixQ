import argparse
from copy import deepcopy
from datetime import datetime
from os import makedirs
from os.path import exists
from typing import List

from torch import bernoulli, device, cuda, no_grad, bool as torch_bool
from torch.nn import Module, ModuleList, BatchNorm1d, CrossEntropyLoss
from torch.nn.functional import dropout
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.transforms import ToUndirected, Compose
from torch_operation_counter import OperationsCounterMode
from numpy import mean, std
from tqdm import tqdm

from quantization.fixed_modules.non_parametric import QReLU
from quantization.fixed_modules.parametric import MaskQuantGraphConvolution
from quantization.mixed_modules.non_parametric import MQReLU
from quantization.mixed_modules.parametric import MQGraphConvolution
from training.logger import setup_logger
from training.probability_degree_transforms import ProbabilisticHighDegreeMask
from utility import flatten_list, format_fraction, nested_median, nested_std


class MQGCN(Module):
    def __init__(self, num_channels: int, hidden_channels: int, out_channels: int, num_bits: List[int],
                 number_of_layers: int):
        super(MQGCN, self).__init__()
        self.number_of_layers = number_of_layers
        self.convs = ModuleList()
        self.relus = ModuleList()
        self.bns = ModuleList()
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
                bn_i = BatchNorm1d(hidden_channels)
                self.relus.append(relu_i)
                self.bns.append(bn_i)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for relu in self.relus:
            relu.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.number_of_layers):
            is_last_layer = i == self.number_of_layers - 1
            x = self.convs[i](x, edge_index, edge_attr)
            if not is_last_layer:
                x = self.relus[i](x)
                x = self.bns[i](x)
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
                x = self.bns[i](x)
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
                x = self.bns[i](x)
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
        self.bns = ModuleList()
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
                bn_i = BatchNorm1d(hidden_channels)
                self.relus.append(relu_i)
                self.bns.append(bn_i)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for relu in self.relus:
            relu.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

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
                x = self.bns[i](x)
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
                x = self.bns[i](x)
                x = dropout(x, p=0.1, training=self.training)
                x = x_in + x
        return x


def train(model, data, train_idx, optimizer, lr_scheduler, criterion, bit_width_lambda=None):
    model.train()
    out = model(data)
    loss = criterion(out[train_idx], data.y.squeeze(1)[train_idx])
    if bit_width_lambda is not None:
        loss = loss + bit_width_lambda * model.calculate_loss(data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    return loss.item()


@no_grad()
def evaluate(model, data, split_idx, evaluator, criterion):
    model.eval()
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({"y_true": data.y[split_idx["train"]],
                                "y_pred": y_pred[split_idx["train"]]})["acc"] * 100
    valid_acc = evaluator.eval({"y_true": data.y[split_idx["valid"]],
                                "y_pred": y_pred[split_idx["valid"]]})["acc"] * 100
    test_acc = evaluator.eval({"y_true": data.y[split_idx["test"]],
                               "y_pred": y_pred[split_idx["test"]]})["acc"] * 100
    train_loss = criterion(out[split_idx["train"]], data.y.squeeze(1)[split_idx["train"]])
    valid_loss = criterion(out[split_idx["valid"]], data.y.squeeze(1)[split_idx["valid"]])
    test_loss = criterion(out[split_idx["test"]], data.y.squeeze(1)[split_idx["test"]])
    return (train_acc, valid_acc, test_acc), (train_loss, valid_loss, test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="ogbn-arxiv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--number_of_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="experimental_plus_DQ_logs")

    parser.add_argument("--bit_width_lambda", type=float, default=-1.0)
    parser.add_argument("--bit_width_search_space", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--lr_quant", type=float, default=0.0001)
    parser.add_argument("--mask_low_probability", type=float, default=0.0)
    parser.add_argument("--mask_high_probability", type=float, default=0.3)
    parser.add_argument("--quant_percentile", type=float, default=0.01)
    parser.add_argument("--quant_use_momentum", type=bool, default=True)
    parser.add_argument("--quant_momentum", type=float, default=0.01)
    args = parser.parse_args()
    arguments = vars(args)

    log_file_name = (f"{args.dataset_name}/"
                     f"hidden_{args.hidden_channels}/"
                     f"wd_{format_fraction(args.weight_decay)}/"
                     f"lr_{format_fraction(args.lr)}/"
                     f"bit_width_{','.join(map(str, args.bit_width_search_space))}"
                     )
    if not exists(args.log_dir + "/" + log_file_name):
        makedirs(args.log_dir + "/" + log_file_name)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name += f"/log_bit_width_lambda_{format_fraction(args.bit_width_lambda)}_{current_time}"
    logger = setup_logger(filename=f"{args.log_dir}/{log_file_name}.log", verbose=True)

    [logger.info(f"{k}: {v}") for k, v in arguments.items()]

    device = device(args.device)

    high_probability = min(args.mask_low_probability + args.mask_high_probability, 1.0)
    mask_transform = ProbabilisticHighDegreeMask(args.mask_low_probability, high_probability)

    dataset = PygNodePropPredDataset(name=args.dataset_name, root="../../data/", transform=Compose([ToUndirected(),
                                                                                                 mask_transform]))
    data = dataset[0].to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].to(device)
    evaluator = Evaluator(name=args.dataset_name)

    criterion = CrossEntropyLoss()

    wining_bit_config, average_bit_widths, quantized_bit_operations, full_precision_accuracies, quantized_accuracies = [], [], [], [], []
    for run in range(args.num_runs):
        logger.info("=" * 100)
        model = MQGCN(num_channels=data.num_node_features,
                      hidden_channels=args.hidden_channels,
                      out_channels=dataset.num_classes,
                      num_bits=args.bit_width_search_space,
                      number_of_layers=args.number_of_layers,
                      ).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        best_state_dict, best_valid_loss, final_test_acc = None, float("inf"), 0
        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            loss = train(model, data, train_idx, optimizer, lr_scheduler, criterion, args.bit_width_lambda)
            accs, losses = evaluate(model, data, split_idx, evaluator, criterion)
            train_acc, valid_acc, test_acc = accs
            train_loss, valid_loss, test_loss = losses
            pbar.set_description(f"Run: {run + 1}/{args.num_runs}, "
                                 f"Loss: {loss:.4f}, "
                                 f"Train: {train_acc:.2f}%, "
                                 f"Valid: {valid_acc:.2f}% "
                                 f"Test: {test_acc:.2f}%")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                final_test_acc = test_acc
                best_state_dict = deepcopy(model.state_dict())

        model.load_state_dict(best_state_dict)
        best_bit_config = model.get_best_bit_config()
        bit_operation = model.estimated_bit_operation_precision(data)
        average_bit_width = mean([*filter(lambda x: x != None, flatten_list(best_bit_config))])

        logger.info(f"Best bit configuration: {best_bit_config}")
        logger.info(f"Bit operations (BOPs): {bit_operation:.2f}")
        logger.info(f"Average bit width: {average_bit_width:.2f}")

        wining_bit_config += [best_bit_config]
        quantized_bit_operations += [bit_operation]
        average_bit_widths += [average_bit_width]

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
        lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        # Full Precision Model Training
        best_state_dict, best_valid_loss, final_test_acc = None, float("inf"), 0
        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            loss = train(model, data, train_idx, optimizer, lr_scheduler, criterion)
            accs, losses = evaluate(model, data, split_idx, evaluator, criterion)
            train_acc, valid_acc, test_acc = accs
            train_loss, valid_loss, test_loss = losses
            pbar.set_description(f"Run: {run + 1}/{args.num_runs}, "
                                 f"Loss: {loss:.4f}, "
                                 f"Train: {train_acc:.2f}%, "
                                 f"Valid: {valid_acc:.2f}% "
                                 f"Test: {test_acc:.2f}%")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                final_test_acc = test_acc
                best_state_dict = deepcopy(model.state_dict())
        logger.info(f"Full precision accuracy: {final_test_acc:.2f}%")
        full_precision_accuracies += [final_test_acc]
        with no_grad():
            with OperationsCounterMode(model) as ops_counter:
                model.full_precision_forward(data)
        full_precision_bit_operations = ops_counter.total_main_operation * 32

        # Quantized Model Training
        model.load_state_dict(best_state_dict)
        model.set_forward_func(model.simulated_quantize_forward)
        optimizer = Adam(model.parameters(), lr=args.lr_quant, weight_decay=args.weight_decay)
        lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        best_valid_acc, final_test_acc = 0, 0
        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            loss = train(model, data, train_idx, optimizer, lr_scheduler, criterion)
            accs, losses = evaluate(model, data, split_idx, evaluator, criterion)
            train_acc, valid_acc, test_acc = accs
            pbar.set_description(f"Run: {run + 1}/{args.num_runs}, "
                                 f"Loss: {loss:.4f}, "
                                 f"Train: {train_acc:.2f}%, "
                                 f"Valid: {valid_acc:.2f}% "
                                 f"Test: {test_acc:.2f}%")
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                final_test_acc = test_acc
        logger.info(f"Average bit width: {average_bit_width:.2f}")
        logger.info(f"Quantized accuracy: {final_test_acc:.2f}%")
        quantized_accuracies += [final_test_acc]

    logger.info("=" * 100)
    logger.info(f"Full precision BOPs: {full_precision_bit_operations:.2f}")
    logger.info(f"Full precision accuracy: {mean(full_precision_accuracies):.2f} ± {std(full_precision_accuracies):.2f}")

    logger.info(f"Average wining bit configuration: {nested_median(*wining_bit_config)}")
    logger.info(f"Standard deviation of wining bit configuration: {nested_std(*wining_bit_config)}")

    logger.info(f"Bit width: {mean(average_bit_widths):.2f} ± {std(average_bit_widths):.2f}")
    logger.info(f"Quantized BOPs: {mean(quantized_bit_operations):.2f} ± {std(quantized_bit_operations):.2f}")
    logger.info(f"Quantized accuracy: {mean(quantized_accuracies):.2f} ± {std(quantized_accuracies):.2f}")
