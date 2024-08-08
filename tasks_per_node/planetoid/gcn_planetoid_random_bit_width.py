from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from torch.nn import Module
from torch.optim import Adam
from torch import cuda, device, no_grad
from torch.nn.functional import cross_entropy, dropout

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from training.tensorboard_logger import TensorboardLogger
from quantization.fixed_modules.non_parametric.activations import QReLU
from quantization.fixed_modules.parametric.graph_convolution import QGraphConvolution


class QGCN(Module):
    def __init__(self, num_channels: int, out_channels: int, num_bits: List[int]):
        super(QGCN, self).__init__()
        self.gcn_1 = QGraphConvolution(in_channels=num_channels,
                                       out_channels=128,
                                       qi=True,
                                       qo=True,
                                       num_bits=num_bits[0],
                                       is_signed=False,
                                       )
        self.relu_1 = QReLU(num_bits=num_bits[1])
        self.gcn_2 = QGraphConvolution(in_channels=128,
                                       out_channels=out_channels,
                                       qi=False,
                                       qo=True,
                                       num_bits=num_bits[2])
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn_1.reset_parameters()
        self.gcn_2.reset_parameters()

    def set_forward_func(self, forward_func: callable):
        self.forward = forward_func

    def full_precision_forward(self, x, edge_index, edge_attr):
        x = self.gcn_1(x, edge_index, edge_attr)
        x = self.relu_1(x)
        x = dropout(x, training=self.training)
        x = self.gcn_2(x, edge_index, edge_attr)
        return x

    def simulated_quantize_forward(self, x, edge_index, edge_attr):
        x = self.gcn_1.simulated_quantize_forward(x, edge_index, edge_attr)
        x = self.relu_1.simulated_quantize_forward(x)
        x = dropout(x, training=self.training)
        x = self.gcn_2.simulated_quantize_forward(x, edge_index, edge_attr)
        return x

    def estimated_bit_operation_precision(self, x, edge_index, edge_attr):
        gcn_1_operations = self.gcn_1.estimated_bit_operation_precision(x, edge_index, edge_attr)
        x = self.gcn_1(x, edge_index, edge_attr)
        relu_bit_operations = self.relu_1.estimated_bit_operation_precision(x)
        x = self.relu_1(x)
        gcn_2_operations = self.gcn_2.estimated_bit_operation_precision(x, edge_index, edge_attr)
        return gcn_1_operations + relu_bit_operations + gcn_2_operations

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)
    accuracies = [(pred[mask] == data.y[mask]).float().mean().item() * 100
                  for mask in (data.train_mask, data.val_mask, data.test_mask)]
    return accuracies


def training_loop(epochs, model, optimizer, data, log_directory="./tensorboard"):
    directory = Path(log_directory) if log_directory is not None else None
    directory.mkdir(parents=True, exist_ok=True) if log_directory is not None else None
    if log_directory is not None:
        tensorboard_logger = TensorboardLogger(log_directory)
    best_val_accuracy, best_epoch, test_accuracy = -float("inf"), 0, 0
    best_state_dict = model.state_dict()
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        loss = train(model, optimizer, data)
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


def g(b=None):
    """
    Randomly sample a bit-width from a list of bit-widths.

    :param b: List of bit-widths to sample from.
    :return: Randomly sampled bit-width.
    """
    if b is None:
        b = [2, 4, 8]
    return np.random.choice(b)


if __name__ == '__main__':
    dataset_name = "Cora"
    num_runs = 30
    epochs = 200
    lr = 0.001
    lr_quant = 0.0001

    device = device("cuda" if cuda.is_available() else "cpu")

    dataset = Planetoid(root="../../data", name=dataset_name, transform=NormalizeFeatures())
    data = dataset[0].to(device)

    fp32_accuracies, simulated_quantize_accuracies, average_bit_width, giga_bit_operations = [], [], [], []
    for run_i in range(num_runs):
        bit_width = [[g(), g(), g(), g(), g(), g()],
                     # Layer 1 bit-widths (Linear input, Linear output, Linear weight), Edge weight, (Message passing input, Message passing output, Message passing weight)
                     [g()],  # Layer 2 bit-widths input
                     [-1, g(), g(), g(), g(), g()],
                     # Layer 2 bit-widths (Linear input, Linear output, Linear weight), Edge weight, (Message passing input, Message passing output, Message passing weight)
                     ]

        model = QGCN(num_channels=dataset.num_features, out_channels=dataset.num_classes, num_bits=bit_width)
        model.to(device)

        model.set_forward_func(model.full_precision_forward)
        optimizer = Adam(model.parameters(), lr=lr)
        best_state_dict, test_accuracy = training_loop(epochs, model, optimizer, data, log_directory=None) # f"./tensorboard/full_precision/{run_i}"
        print(f"[{run_i + 1}/{num_runs}]: FP32 Model, Accuracy: {test_accuracy:.2f}%")
        fp32_accuracies.append(test_accuracy)

        model.load_state_dict(best_state_dict)
        model.set_forward_func(model.simulated_quantize_forward)
        optimizer = Adam(model.parameters(), lr=lr_quant)
        best_state_dict, test_accuracy = training_loop(epochs, model, optimizer, data, log_directory=None) # f"./tensorboard/simulated_quantize/{run_i}"
        print(f"[{run_i + 1}/{num_runs}]: Simulated Quantized Model, Accuracy: {test_accuracy:.2f}%")
        simulated_quantize_accuracies.append(test_accuracy)
        average_bit_width.append(np.mean([bit for layer in bit_width for bit in layer if bit > 0]))
        giga_bit_operations.append(model.estimated_bit_operation_precision(data.x, data.edge_index, data.edge_attr) / 1e9)

    print("=" * 80)
    print(f"FP32 Model, Accuracy: {np.mean(fp32_accuracies):.2f}% ± {np.std(fp32_accuracies):.2f}%")
    print(f"Average Bit Width: {np.mean(average_bit_width):.2f} ± {np.std(average_bit_width):.2f}")
    print(f"Average Giga Bit Operations: {np.mean(giga_bit_operations):.2f} ± {np.std(giga_bit_operations):.2f}")
    print(f"Quantized Model, Accuracy: {np.mean(simulated_quantize_accuracies):.2f}% ± {np.std(simulated_quantize_accuracies):.2f}%")
