import random
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from tqdm import tqdm

from torch.nn import Module, ModuleList
from torch.optim import lr_scheduler, Adam
from torch import cat, device, no_grad, cuda, tensor, save, load
from torch.nn.functional import cross_entropy, dropout

from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.nn import global_max_pool
from torch_geometric.loader import DataLoader

from quantization.fixed_modules.non_parametric.activations import QReLU
from quantization.fixed_modules.parametric.graph_isomorphism import QGraphIsomorphism
from quantization.fixed_modules.parametric.linear import QLinear, QLinearBatchNormReLU


def setup_seed(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class QTwoLinearBNReLU(Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 qi,
                 qo,
                 num_bits,
                 dropout_p=0.2,
                 ):
        super(QTwoLinearBNReLU, self).__init__()
        self.dropout_p = dropout_p
        self.lin_bn_relu_1 = QLinearBatchNormReLU(in_features=in_channels,
                                                  out_features=hidden_channels,
                                                  qi=qi,
                                                  qo=True,
                                                  num_bits=num_bits,
                                                  )
        self.lin_bn_relu_2 = QLinearBatchNormReLU(in_features=hidden_channels,
                                                  out_features=out_channels,
                                                  qi=False,
                                                  qo=qo,
                                                  num_bits=num_bits,
                                                  )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_bn_relu_1.reset_parameters()
        self.lin_bn_relu_2.reset_parameters()

    def set_forward_func(self, forward_func: callable):
        self.forward = forward_func

    def full_precision_forward(self, x):
        x = self.lin_bn_relu_1(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin_bn_relu_2(x)
        return x

    def simulated_quantize_forward(self, x):
        x = self.lin_bn_relu_1.simulated_quantize_forward(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin_bn_relu_2.simulated_quantize_forward(x)
        return x

    def freeze(self, qi=None):
        self.lin_bn_relu_1.freeze(qi=qi)
        self.lin_bn_relu_2.freeze(qi=self.lin_bn_relu_1.qo)
        self.qi = self.lin_bn_relu_1.qi.copy()
        self.qo = self.lin_bn_relu_2.qo.copy()

    def quantize_inference(self, x):
        qx = self.lin_bn_relu_1.quantize_inference(x)
        qx = self.lin_bn_relu_2.quantize_inference(qx)
        return qx



class QGIN(Module):
    def __init__(self,
                 in_channels,
                 num_layers,
                 hidden_channels,
                 out_channels,
                 bit_widths,
                 dropout_p=0.2,
                 ):
        super(QGIN, self).__init__()

        self.dropout_p = dropout_p
        self.embedding = QLinear(in_features=in_channels,
                                 out_features=hidden_channels,
                                 qi=True,
                                 qo=True,
                                 num_bits=bit_widths[0],
                                 )
        self.convs = ModuleList()
        for i in range(num_layers):
            mlp_i = QTwoLinearBNReLU(hidden_channels,
                                     hidden_channels,
                                     hidden_channels,
                                     qi=False,
                                     qo=True,
                                     num_bits=bit_widths[i + 1][0],
                                     dropout_p=dropout_p,
                                     )
            gin_i = QGraphIsomorphism(nn=mlp_i, qi=False, qo=True, num_bits=bit_widths[i + 1][1])
            self.convs.append(gin_i)

        self.lin1 = QLinear(in_features=hidden_channels,
                            out_features=hidden_channels,
                            qi=True,
                            qo=True,
                            num_bits=bit_widths[num_layers + 1],
                            )
        self.relu = QReLU(num_bits=bit_widths[num_layers + 2])
        self.lin2 = QLinear(in_features=hidden_channels,
                            out_features=out_channels,
                            qi=False,
                            qo=True,
                            num_bits=bit_widths[num_layers + 3],
                            )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def set_forward_func(self, forward_func: callable):
        self.forward = forward_func

    def full_precision_forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = cat((x, pos), dim=1)
        x = self.embedding(x)
        for conv in self.convs:
            x = dropout(x, p=self.dropout_p, training=self.training)
            x = conv(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

    def simulated_quantize_forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = cat((x, pos), dim=1)
        x = self.embedding.simulated_quantize_forward(x)
        for conv in self.convs:
            x = dropout(x, p=self.dropout_p, training=self.training)
            x = conv.simulated_quantize_forward(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.lin1.simulated_quantize_forward(x)
        x = self.relu.simulated_quantize_forward(x)
        x = self.lin2.simulated_quantize_forward(x)
        return x

    def freeze(self):
        self.embedding.freeze()
        qo = self.embedding.qo
        for conv in self.convs:
            conv.freeze(qi=qo)
            qo = conv.qo
        self.lin1.freeze(qi=qo)
        self.relu.freeze(qi=self.lin1.qo)
        self.lin2.freeze(qi=self.lin1.qo)

    def quantize_inference(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        qx = cat((x, pos), dim=1)
        qx = self.embedding.qi.quantize(qx)
        qx = self.embedding.quantize_inference(qx)
        for conv in self.convs:
            qx = conv.quantize_inference(qx, edge_index)
        qx = global_max_pool(qx, batch)
        qx = self.lin1.quantize_inference(qx)
        qx = self.relu.quantize_inference(qx)
        qx = self.lin2.quantize_inference(qx)
        x = self.lin2.qo.dequantize(qx)
        return x


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = cross_entropy(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_accuracy(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset) * 100


def eval_loss(model, loader):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        with no_grad():
            out = model(data)
        loss += cross_entropy(out, data.y.view(-1), reduction="sum").item()
    return loss / len(loader.dataset)


if __name__ == '__main__':
    setup_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--max_cycle", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_quant", type=float, default=0.00001)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_schedule_patience", type=int, default=10)
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--num_bits", type=int, default=8)
    args = parser.parse_args()

    train_dataset = GNNBenchmarkDataset(root="../data", name=args.dataset_name, split="train")
    val_dataset = GNNBenchmarkDataset(root="../data", name=args.dataset_name, split="val")
    test_dataset = GNNBenchmarkDataset(root="../data", name=args.dataset_name, split="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = device("cuda" if cuda.is_available() else "cpu")

    bit_widths = [[args.num_bits, args.num_bits, args.num_bits]]
    bit_widths += [[[-1, args.num_bits, args.num_bits],  # MLP Bit-widths
                   [-1, args.num_bits]  # Graph Isomorphism Bit-widths
                   ] for _ in range(args.num_layers)]
    bit_widths += [[args.num_bits, args.num_bits, args.num_bits],
                   [args.num_bits],
                   [args.num_bits, args.num_bits, args.num_bits],
                   ]

    full_precision_accuracies = []
    simulated_accuracies = []
    quantized_accuracies = []
    for k in range(1):
        for i in range(args.max_cycle):
            model = QGIN(in_channels=train_dataset[0].x.size()[1] + train_dataset[0].pos.size()[1],
                         hidden_channels=args.hidden_units,
                         out_channels=train_dataset.num_classes,
                         num_layers=args.num_layers,
                         dropout_p=args.dropout_p,
                         bit_widths=bit_widths,
                         ).to(device)
            model.set_forward_func(model.full_precision_forward)
            [conv_i.nn.set_forward_func(conv_i.nn.full_precision_forward) for conv_i in model.convs]

            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode="max",
                                                       threshold=1e-4,
                                                       factor=args.lr_decay_factor,
                                                       patience=args.lr_schedule_patience,
                                                       )
            best_epoch, best_val_accuracy, best_state_dict = 0, 0, None
            pbar = tqdm(range(args.max_epoch))
            for epoch in pbar:
                train_loss = train(model, optimizer, train_loader)
                val_accuracy = eval_accuracy(model, val_loader)
                scheduler.step(val_accuracy)
                pbar.set_description("Loss: {:.3f}, "
                                     "Val Acc.: {:.3f}, "
                                     "lr: {:.6f}, "
                                     "Best Val Acc.: {:.3f}, ".format(train_loss,
                                                                      val_accuracy,
                                                                      scheduler.get_last_lr()[0],
                                                                      best_val_accuracy))
                if val_accuracy > best_val_accuracy:
                    best_epoch = epoch
                    best_val_accuracy = val_accuracy
                    best_state_dict = deepcopy(model.state_dict().copy())

            print("Best Epoch: {:03d}, Best Val Loss: {:.3f}".format(best_epoch, best_val_accuracy))
            model.load_state_dict(best_state_dict)
            model.eval()

            test_accuracy = eval_accuracy(model, test_loader)
            full_precision_accuracies.append(test_accuracy)

            model.set_forward_func(model.simulated_quantize_forward)
            [conv_i.nn.set_forward_func(conv_i.nn.simulated_quantize_forward) for conv_i in model.convs]

            with no_grad():
                sample = next(iter(DataLoader(train_dataset, batch_size=16384, shuffle=True)))
                model.simulated_quantize_forward(sample.to(device))
            optimizer = Adam(model.parameters(), lr=args.lr_quant, weight_decay=args.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode="max",
                                                       threshold=1e-8,
                                                       factor=args.lr_decay_factor,
                                                       patience=args.lr_schedule_patience,
                                                       )
            best_epoch, best_val_accuracy, best_state_dict = 0, 0, None
            pbar = tqdm(range(args.max_epoch))
            for epoch in pbar:
                train_loss = train(model, optimizer, train_loader)
                val_accuracy = eval_accuracy(model, val_loader)
                scheduler.step(val_accuracy)
                pbar.set_description("Loss: {:.3f}, "
                                     "Val Acc.: {:.3f}, "
                                     "lr: {:.6f}, "
                                     "Best Val Accu.: {:.3f}, ".format(train_loss,
                                                                       val_accuracy,
                                                                       scheduler.get_last_lr()[0],
                                                                       best_val_accuracy))
                if val_accuracy > best_val_accuracy:
                    best_epoch = epoch
                    best_val_accuracy = val_accuracy
                    best_state_dict = deepcopy(model.state_dict().copy())

            print("Best Epoch: {:03d}, Best Val Accuracy: {:.3f}".format(best_epoch, best_val_accuracy))
            model.load_state_dict(best_state_dict)
            model.eval()

            test_accuracy = eval_accuracy(model, test_loader)
            simulated_accuracies.append(test_accuracy)

            model.freeze()
            model.set_forward_func(model.quantize_inference)
            [conv_i.nn.set_forward_func(conv_i.nn.quantize_inference) for conv_i in model.convs]

            test_accuracy = eval_accuracy(model, test_loader)
            quantized_accuracies.append(test_accuracy)

        full_precision_accuracies = tensor(full_precision_accuracies)
        simulated_accuracies = tensor(simulated_accuracies)
        quantized_accuracies = tensor(quantized_accuracies)

        full_precision_acc_mean = full_precision_accuracies.mean()
        full_precision_acc_std = full_precision_accuracies.std()
        full_precision_desc = "{:.3f} ± {:.3f}".format(full_precision_acc_mean, full_precision_acc_std)
        print("Result - {}".format(full_precision_desc))

        simulated_acc_mean = simulated_accuracies.mean()
        simulated_acc_std = simulated_accuracies.std()
        simulated_desc = "{:.3f} ± {:.3f}".format(simulated_acc_mean, simulated_acc_std)
        print("Result - {}".format(simulated_desc))

        quantized_acc_mean = quantized_accuracies.mean()
        quantized_acc_std = quantized_accuracies.std()
        quantized_desc = "{:.3f} ± {:.3f}".format(quantized_acc_mean, quantized_acc_std)
        print("Result - {}".format(quantized_desc))
