import argparse
from copy import deepcopy
from datetime import datetime
from os import makedirs
from os.path import exists
from typing import List

from numpy import mean, std
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from torch.optim import Adam
from torch.nn import ModuleList, Module, BatchNorm1d, BCEWithLogitsLoss
from torch.nn.functional import dropout, binary_cross_entropy
from torch import no_grad, cat, cuda, device as torch_device, save, load, float32
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import index_to_mask
from torch_geometric.transforms import Compose, ToDevice, AddSelfLoops, AddLaplacianEigenvectorPE
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from quantization.fixed_modules.non_parametric import QReLU
from quantization.fixed_modules.parametric import QLinear
from quantization.mixed_modules.non_parametric import MQReLU
from quantization.mixed_modules.parametric import MQLinear
from training.logger import setup_logger
from utility import format_fraction, flatten_list

from rebuttal_phase.SAGEConv import MQGraphSAGE, QGraphSAGE


class MQGCN(Module):
    def __init__(self, num_channels: int, hidden_channels: int, out_channels: int, num_bits: List[int],
                 number_of_layers: int):
        super(MQGCN, self).__init__()
        self.number_of_layers = number_of_layers
        self.lin1 = MQLinear(in_features=num_channels,
                             out_features=hidden_channels,
                             qi=True,
                             qo=False,
                             num_bits_list=num_bits,
                             is_signed=False,
                             quantize_per="column")
        self.convs = ModuleList()
        self.relus = ModuleList()
        self.bns = ModuleList()
        for i in range(number_of_layers):
            gcn_i = MQGraphSAGE(in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                qi=False,
                                qo=True,
                                num_bits_list=num_bits,
                                is_signed=False,
                                quantize_per="column",
                                )
            relu_i = MQReLU(num_bits_list=num_bits, quantize_per="column")
            bn_i = BatchNorm1d(hidden_channels)
            self.convs.append(gcn_i)
            self.relus.append(relu_i)
            self.bns.append(bn_i)

        self.lin2 = MQLinear(in_features=hidden_channels,
                             out_features=out_channels,
                             qi=False,
                             qo=True,
                             num_bits_list=num_bits,
                             is_signed=False,
                             quantize_per="column")
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for relu in self.relus:
            relu.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x_local = 0
        x = self.lin1(x)
        x_local += x
        for i in range(self.number_of_layers):
            x = self.relus[i](x)
            x = dropout(x, p=0.5, training=self.training)
            x = self.convs[i](x, edge_index)
            x_local += x
            x = self.bns[i](x)
        x = dropout(x, p=0.5, training=self.training)
        return self.lin2(x)

    def calculate_loss(self, x, edge_index):
        loss = 0
        loss += self.lin1.calculate_weighted_loss(x)
        x = self.lin1(x)
        for i in range(self.number_of_layers):
            loss += self.relus[i].calculate_weighted_loss(x)
            x = self.relus[i](x)
            loss += self.convs[i].calculate_weighted_loss(x, edge_index)
            x = self.convs[i](x, edge_index)
        loss += self.lin2.calculate_weighted_loss(x)
        return loss

    @no_grad()
    def estimated_bit_operation_precision(self, x, edge_index):
        bit_operations = 0
        bit_operations += self.lin1.estimated_bit_operation_precision(x)
        x = self.lin1(x)
        for i in range(self.number_of_layers):
            bit_operations += self.relus[i].estimated_bit_operation_precision(x)
            x = self.relus[i](x)
            bit_operations += self.convs[i].estimated_bit_operation_precision(x, edge_index)
            x = self.convs[i](x, edge_index)
        bit_operations += self.lin2.estimated_bit_operation_precision(x)
        return bit_operations

    def get_best_bit_config(self):
        bit_config = []
        bit_config += [sum(self.lin1.select_top_k_winners(1).values(), []), ]
        for i in range(self.number_of_layers):
            conv_i_bit_config = sum(self.convs[i].select_top_k_winners(1).values(), [])
            relu_i_bit_config = sum(self.relus[i].select_top_k_winners(1).values(), [])
            bit_config += [[conv_i_bit_config, relu_i_bit_config], ]
        bit_config += [sum(self.lin2.select_top_k_winners(1).values(), []), ]
        return bit_config


class QGCN(Module):
    def __init__(self, num_channels: int, hidden_channels: int, out_channels: int, num_bits: List[int],
                 number_of_layers: List[List[int]]):
        super(QGCN, self).__init__()
        self.number_of_layers = number_of_layers
        self.lin1 = QLinear(in_features=num_channels,
                            out_features=hidden_channels,
                            qi=True,
                            qo=False,
                            num_bits=num_bits[0],
                            is_signed=False,
                            quantize_per="column")
        self.convs = ModuleList()
        self.relus = ModuleList()
        self.bns = ModuleList()
        i = 0
        for i in range(number_of_layers):
            gcn_i = QGraphSAGE(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               qi=False,
                               qo=True,
                               num_bits=num_bits[i + 1][0],
                               is_signed=False,
                               quantize_per="column",
                               )
            relu_i = QReLU(num_bits=num_bits[i + 1][1], quantize_per="column")
            bn_i = BatchNorm1d(hidden_channels)
            self.convs.append(gcn_i)
            self.relus.append(relu_i)
            self.bns.append(bn_i)
        self.lin2 = QLinear(in_features=hidden_channels,
                            out_features=out_channels,
                            qi=False,
                            qo=True,
                            num_bits=num_bits[i + 2],
                            is_signed=False,
                            quantize_per="column")
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for relu in self.relus:
            relu.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin2.reset_parameters()

    def set_forward_func(self, forward_func: callable):
        self.forward = forward_func

    def full_precision_forward(self, x, edge_index):
        x_local = 0
        x = self.lin1(x)
        x_local += x
        for i in range(self.number_of_layers):
            x = self.relus[i](x)
            x = dropout(x, p=0.5, training=self.training)
            x = self.convs[i](x, edge_index)
            x_local += x
            x = self.bns[i](x)
        x = dropout(x, p=0.5, training=self.training)
        return self.lin2(x)

    def simulated_quantize_forward(self, x, edge_index):
        x_local = 0
        x = self.lin1.simulated_quantize_forward(x)
        x_local += x
        for i in range(self.number_of_layers):
            x = self.relus[i].simulated_quantize_forward(x)
            x = dropout(x, p=0.5, training=self.training)
            x = self.convs[i].simulated_quantize_forward(x, edge_index)
            x_local += x
            x = self.bns[i](x)
        x = dropout(x, p=0.5, training=self.training)
        return self.lin2.simulated_quantize_forward(x)


def train(model, loader, optimizer, loss_func, transform, bit_width_lambda=None):
    model.train()
    total_loss = total_examples = 0
    for data in loader:
        optimizer.zero_grad()
        data = transform(data)
        out = model(data.x, data.edge_index)
        loss = loss_func(out[data.train_mask], data.y[data.train_mask].to(float32))
        if bit_width_lambda is not None:
            loss = loss + bit_width_lambda * model.calculate_loss(data.x, data.edge_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())
    return total_loss / total_examples


@no_grad()
def evaluate(model, loader, evaluator, transform):
    model.eval()
    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}
    for data in loader:
        data = transform(data)
        out = model(data.x, data.edge_index)
        for split in ["train", "valid", "test"]:
            mask = data[f"{split}_mask"]
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())
    train_rocauc = evaluator.eval({"y_true": cat(y_true["train"], dim=0),
                                   "y_pred": cat(y_pred["train"], dim=0),
                                   })["rocauc"]
    valid_rocauc = evaluator.eval({"y_true": cat(y_true["valid"], dim=0),
                                   "y_pred": cat(y_pred["valid"], dim=0),
                                   })["rocauc"]
    test_rocauc = evaluator.eval({"y_true": cat(y_true["test"], dim=0),
                                  "y_pred": cat(y_pred["test"], dim=0),
                                  })["rocauc"]
    return train_rocauc, valid_rocauc, test_rocauc


def training_loop(epoch, model, optimizer, scheduler, train_loader, test_loader, evaluator, transform, eval_step=10, bit_width_lambda=None):
    best_state_dict, final_train_rocauc, best_val_rocauc, final_test_rocauc = None, 0.0, 0.0, 0.0
    loss_func = BCEWithLogitsLoss()
    pbar = tqdm(range(epoch))
    for epoch in pbar:
        loss = train(model, train_loader, optimizer, loss_func, transform, bit_width_lambda)
        if epoch % eval_step == 0:
            train_acc, val_acc, test_acc = evaluate(model, test_loader, evaluator, transform)
            if val_acc > best_val_rocauc:
                best_val_rocauc = val_acc
                final_test_rocauc = test_acc
                best_state_dict = deepcopy(model.state_dict().copy())
        if scheduler is not None:
            scheduler.step()
        pbar.set_description((f"Epoch: {epoch:02d}, "
                              f"Loss: {loss:.4f}, "
                              f"Train: {train_acc:.2f}, "
                              f"Valid: {val_acc:.2f}, "
                              f"Test: {test_acc:.2f}, "
                              f"Best Valid: {best_val_rocauc:.2f}, "
                              f"Best Test: {final_test_rocauc:.2f}"))
    return final_test_rocauc, best_state_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="ogbn-proteins")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--number_of_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_quant", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="experimental_logs")

    parser.add_argument("--bit_width_lambda", type=float, default=1.0)
    parser.add_argument("--bit_width_search_space", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()
    arguments = vars(args)

    log_path = (f"dataset_{args.dataset_name}/"
                f"GraphSAGE/"
                f"hidden_{args.hidden_channels}/"
                f"wd_{format_fraction(args.weight_decay)}/"
                f"lr_{format_fraction(args.lr)}/"
                f"bit_width_{','.join(map(str, args.bit_width_search_space))}"
                )
    if not exists(args.log_dir + "/" + log_path):
        makedirs(args.log_dir + "/" + log_path)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name = log_path + f"/log_bit_width_lambda_{format_fraction(args.bit_width_lambda)}_{current_time}"
    relaxed_model_file_path = f"{args.log_dir}/{log_path}/relaxed_architecture_bit_width_lambda_{format_fraction(args.bit_width_lambda)}.pt"
    fp32_model_file_path = f"{args.log_dir}/{log_path}/fp32_architecture.pt"
    logger = setup_logger(filename=f"{args.log_dir}/{log_file_name}.log", verbose=True)

    [logger.info(f"{k}: {v}") for k, v in arguments.items()]
    logger.info("=" * 100)

    device = torch_device(args.device)

    transform = Compose([ToDevice(device)])
    dataset = PygNodePropPredDataset(args.dataset_name, "../data", transform=Compose([AddSelfLoops(),
                                                                                      AddLaplacianEigenvectorPE(k=16, attr_name="x")]))
    evaluator = Evaluator(name=args.dataset_name)

    data = dataset[0]
    split_idx = dataset.get_idx_split()
    for split in ["train", "valid", "test"]:
        data[f"{split}_mask"] = index_to_mask(split_idx[split], data.y.shape[0])

    train_loader = RandomNodeLoader(data, num_parts=2, shuffle=True, num_workers=1)
    # Increase the num_parts of the test loader if you cannot fit
    test_loader = RandomNodeLoader(data, num_parts=2, num_workers=1)

    wining_bit_config, average_bit_widths, quantized_bit_operations, full_precision_accuracies, quantized_accuracies = [], [], [], [], []
    model = MQGCN(
        num_channels=16,
        hidden_channels=args.hidden_channels,
        out_channels=112,
        number_of_layers=args.number_of_layers,
        num_bits=args.bit_width_search_space,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.1)

    if not exists(relaxed_model_file_path):
        test_rocauc, state_dict = training_loop(10, model, optimizer, None, train_loader, test_loader, evaluator, transform, 2, args.bit_width_lambda)
        model.load_state_dict(state_dict)
        best_bit_config = model.get_best_bit_config()
        bit_operation = model.cpu().estimated_bit_operation_precision(data.x, data.edge_index)
        average_bit_width = mean([*filter(lambda x: x != None, flatten_list(best_bit_config))])
        save({"state_dict": state_dict,
              "test_rocauc": test_rocauc,
              "best_bit_config": best_bit_config,
              "bit_operation": bit_operation,
              "average_bit_width": average_bit_width,
              }, relaxed_model_file_path)
    else:
        state_dict = load(relaxed_model_file_path, weights_only=False)["state_dict"]
        test_rocauc = load(relaxed_model_file_path)["test_rocauc"]
        best_bit_config = load(relaxed_model_file_path)["best_bit_config"]
        bit_operation = load(relaxed_model_file_path)["bit_operation"]
        average_bit_width = load(relaxed_model_file_path)["average_bit_width"]

    wining_bit_config += [best_bit_config]
    quantized_bit_operations += [bit_operation]
    average_bit_widths += [average_bit_width]

    logger.info(f"Best bit configuration: {best_bit_config}")
    logger.info(f"Bit operations (BOPs): {bit_operation:.2f}")
    logger.info(f"Average bit width: {average_bit_width:.2f}")
    logger.info(f"Relaxed test rocauc: {test_rocauc:.2f}")

    model = QGCN(
        num_channels=16,
        hidden_channels=args.hidden_channels,
        out_channels=112,
        number_of_layers=args.number_of_layers,
        num_bits=best_bit_config,
    ).to(device)
    model.set_forward_func(model.full_precision_forward)
    optimizer = Adam(model.parameters(), lr=args.lr)

    if not exists(fp32_model_file_path):
        test_rocauc, state_dict = training_loop(5 * args.epochs, model, optimizer, None, train_loader, test_loader, evaluator, transform)
        save({"state_dict": state_dict, "test_rocauc": test_rocauc}, fp32_model_file_path)
    else:
        state_dict = load(fp32_model_file_path, weights_only=False)["state_dict"]
        test_rocauc = load(fp32_model_file_path)["test_rocauc"]

    logger.info(f"Full precision test rocauc: {test_rocauc:.2f}")

    full_precision_accuracies += [test_rocauc]

    for run in range(args.num_runs):
        model.reset_parameters()
        model.load_state_dict(state_dict)
        model.set_forward_func(model.simulated_quantize_forward)
        model.cpu()(data.x, data.edge_index)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr_quant)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        test_rocauc, _ = training_loop(args.epochs, model, optimizer, scheduler, train_loader, test_loader, evaluator, transform)
        logger.info(f"Quantized test rocauc: {test_rocauc:.2f}")
        quantized_accuracies += [test_rocauc]

    logger.info("=" * 100)
    logger.info(f"Average bit width: {mean(average_bit_widths):.2f}")
    logger.info(f"Bit operations (BOPs): {mean(quantized_bit_operations):.2f}")
    logger.info(f"Full precision test rocauc: {mean(full_precision_accuracies):.2f}")
    logger.info(f"Quantized test rocauc: {mean(quantized_accuracies):.2f} ± {std(quantized_accuracies):.2f}")
    logger.info("=" * 100)
