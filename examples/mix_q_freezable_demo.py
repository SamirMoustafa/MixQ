from typing import List
import matplotlib.pyplot as plt

from torch.nn import Module
from torch.optim import Adam
from torch import cuda, device, no_grad, tensor
from torch.nn.functional import cross_entropy

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from quantization.fixed_modules.parametric.graph_convolution import QGraphConvolution
from quantization.mixed_modules.parametric import MQGraphConvolution


class MQGCN(Module):
    def __init__(self, num_channels: int, out_channels: int, num_bits: List[int]):
        super(MQGCN, self).__init__()
        self.gcn_1 = MQGraphConvolution(in_channels=num_channels,
                                        out_channels=out_channels,
                                        qi=True,
                                        qo=True,
                                        num_bits_list=num_bits,
                                        is_signed=False,
                                        )
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn_1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.gcn_1(x, edge_index, edge_attr)
        return x

    def calculate_loss(self, x, edge_index, edge_attr):
        loss = 0
        loss += self.gcn_1.calculate_weighted_loss(x, edge_index, edge_attr)
        x = self.gcn_1(x, edge_index, edge_attr)
        return loss


class QGCN(Module):
    def __init__(self, num_channels: int, out_channels: int, num_bits: List[int]):
        super(QGCN, self).__init__()
        self.gcn_1 = QGraphConvolution(in_channels=num_channels,
                                       out_channels=out_channels,
                                       qi=True,
                                       qo=True,
                                       num_bits=num_bits[0],
                                       is_signed=False,
                                       )
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn_1.reset_parameters()

    def set_forward_func(self, forward_func: callable):
        self.forward = forward_func

    def full_precision_forward(self, x, edge_index, edge_attr):
        x = self.gcn_1(x, edge_index, edge_attr)
        return x

    def simulated_quantize_forward(self, x, edge_index, edge_attr):
        x = self.gcn_1.simulated_quantize_forward(x, edge_index, edge_attr)
        return x

    def freeze(self):
        self.gcn_1.freeze()

    def quantize_inference(self, x, edge_index, edge_attr):
        qx = self.gcn_1.qi.quantize(x)
        qx = self.gcn_1.quantize_inference(qx, edge_index, edge_attr)
        x = self.gcn_1.qo.dequantize(qx)
        return x


def train(model, optimizer, data, bit_width_lambda=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    classification_loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
    if bit_width_lambda is not None:
        bit_width_loss = bit_width_lambda * model.calculate_loss(data.x, data.edge_index, data.edge_attr)
    else:
        bit_width_loss = tensor([0.0], device=out.device)
    (classification_loss + bit_width_loss).backward()
    optimizer.step()
    return classification_loss.item()


@no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)
    accuracies = [(pred[mask] == data.y[mask]).float().mean().item() * 100
                  for mask in (data.train_mask, data.val_mask, data.test_mask)]
    return accuracies


def training_loop(epochs, model, optimizer, data, bit_width_lambda=None):
    best_val_accuracy, best_epoch, test_accuracy = -float("inf"), 0, 0
    best_state_dict = model.state_dict()
    for epoch in range(1, epochs + 1):
        train(model, optimizer, data, bit_width_lambda)
        train_accuracy, validation_accuracy, tmp_test_accuracy = evaluate(model, data)
        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            test_accuracy = tmp_test_accuracy
            best_state_dict = model.state_dict().copy()
    return best_state_dict, test_accuracy


if __name__ == '__main__':
    num_bits_list = [2, 4, 8, 16, 32]
    epochs = 500
    lr = 1e-1
    lr_mix = 1e-2
    lr_quant = 1e-5
    bit_width_lambda = 1e-8

    device = device("cuda" if cuda.is_available() else "cpu")

    dataset = Planetoid(root="../data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)
    # Train over the entire dataset to neglect the effect of the train/val/test split
    data.train_mask.fill_(1)
    data.val_mask.fill_(1)
    data.test_mask.fill_(1)

    # Define relaxed model to find the winning bit width (MixQ core model)
    model = MQGCN(dataset.num_features, dataset.num_classes, num_bits_list)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr_mix)
    best_state_dict, test_accuracy = training_loop(epochs, model, optimizer, data, bit_width_lambda)
    wining_bit_width = [sum(model.gcn_1.select_top_k_winners(1).values(), []), ]

    # Define the quantizable model with the winning bit width
    model = QGCN(num_channels=dataset.num_features, out_channels=dataset.num_classes, num_bits=wining_bit_width)
    model.to(device)

    # Train the full-precision model
    model.set_forward_func(model.full_precision_forward)
    optimizer = Adam(model.parameters(), lr=lr)
    best_state_dict, test_accuracy = training_loop(epochs, model, optimizer, data)
    fp32_accuracy = test_accuracy
    fp32_learnable_weights = model.gcn_1.lin.linear_module.weight.clone().detach()
    model.load_state_dict(best_state_dict)

    # Train the quantizable model with the winning bit width
    model.set_forward_func(model.simulated_quantize_forward)
    optimizer = Adam(model.parameters(), lr=lr_quant)
    best_state_dict, test_accuracy = training_loop(epochs, model, optimizer, data)
    simulated_quantize_accuracy = test_accuracy
    simulated_quantize_learnable_weights = model.gcn_1.lin.linear_module.weight.clone().detach()

    # Load the best state dict and evaluate the quantized inference model after fusion
    model.load_state_dict(best_state_dict)
    model.eval()
    model.freeze()
    model.set_forward_func(model.quantize_inference)
    _, _, test_accuracy = evaluate(model, data)
    quantize_inference_accuracy = test_accuracy
    quantize_inference_learnable_weights = model.gcn_1.lin.quantized_weight.clone().detach()

    print("=" * 80)
    print(f"Winning bit width: {wining_bit_width}")
    print(f"Full-precision model accuracy: {fp32_accuracy:.1f}%")
    print(f"Simulated quantized model accuracy: {simulated_quantize_accuracy:.1f}%")
    print(f"Quantized inference model accuracy: {quantize_inference_accuracy:.1f}%")

    # Plot the learnable weights of the full-precision, simulated quantized, and quantized inference models
    def plot_tensor_histogram(tensor, title):
        plt.figure()
        plt.text(0.05, 0.95, f"Mean: {tensor.mean().item():.2f}", transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f"Std: {tensor.std().item():.2f}", transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f"Max: {tensor.max().item():.2f}", transform=plt.gca().transAxes)
        plt.text(0.05, 0.80, f"Min: {tensor.min().item():.2f}", transform=plt.gca().transAxes)
        plt.text(0.05, 0.75, f"Unique: {len(tensor.unique()):.0f}", transform=plt.gca().transAxes)
        plt.hist(tensor.flatten().cpu().numpy(), bins=500, color="#0065a7")
        plt.title(title)
        plt.grid(axis="x")
        plt.tight_layout()
        plt.show()

    plot_tensor_histogram(fp32_learnable_weights, "Full-precision model learnable weights")
    plot_tensor_histogram(simulated_quantize_learnable_weights, "Simulated quantized model learnable weights")
    plot_tensor_histogram(quantize_inference_learnable_weights, "Quantized inference model learnable weights")

