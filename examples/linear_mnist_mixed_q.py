from copy import deepcopy

from math import prod
from tqdm import tqdm

from torch.optim import AdamW
from torch import cuda, device, tensor
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from quantization.fixed_modules.parametric.linear import QLinear
from quantization.fixed_modules.non_parametric.activations import QReLU
from quantization.mixed_modules.parametric.linear import MQLinear
from quantization.mixed_modules.non_parametric.activations import MQReLU


class MQNetwork(Module):
    def __init__(self, num_channels, out_channels, num_bits):
        super(MQNetwork, self).__init__()
        self.linear_1 = MQLinear(in_features=num_channels,
                                 out_features=512,
                                 qi=True,
                                 qo=True,
                                 num_bits_list=num_bits, )
        self.relu_1 = MQReLU(qi=False, num_bits_list=num_bits)
        self.linear_2 = MQLinear(in_features=512,
                                 out_features=256,
                                 qi=False,
                                 qo=True,
                                 num_bits_list=num_bits, )
        self.relu_2 = MQReLU(qi=False, num_bits_list=num_bits)
        self.linear_3 = MQLinear(in_features=256,
                                 out_features=out_channels,
                                 qi=False,
                                 qo=True,
                                 num_bits_list=num_bits, )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.linear_3(x)
        return x

    def calculate_loss(self, x):
        x = x.view(x.size(0), -1)
        loss = self.linear_1.calculate_weighted_loss(x)
        x = self.linear_1(x)
        loss += self.relu_1.calculate_weighted_loss(x)
        x = self.relu_1(x)
        loss += self.linear_2.calculate_weighted_loss(x)
        x = self.linear_2(x)
        loss += self.relu_2.calculate_weighted_loss(x)
        x = self.relu_2(x)
        loss += self.linear_3.calculate_weighted_loss(x)
        return loss

    def estimated_bit_operation_precision(self, x):
        integer_operations = 0
        x = x.view(x.size(0), -1)
        integer_operations += self.linear_1.estimated_bit_operation_precision(x)
        x = self.linear_1(x)
        integer_operations += self.relu_1.estimated_bit_operation_precision(x)
        x = self.relu_1(x)
        integer_operations += self.linear_2.estimated_bit_operation_precision(x)
        x = self.linear_2(x)
        integer_operations += self.relu_2.estimated_bit_operation_precision(x)
        x = self.relu_2(x)
        integer_operations += self.linear_3.estimated_bit_operation_precision(x)
        return integer_operations


class QNetwork(Module):
    def __init__(self, num_channels, out_channels, num_bits):
        super(QNetwork, self).__init__()
        self.linear_1 = QLinear(in_features=num_channels,
                                out_features=512,
                                qi=True,
                                qo=True,
                                num_bits=num_bits[0],
                                )
        self.relu_1 = QReLU(qi=True, num_bits=num_bits[1])
        self.linear_2 = QLinear(in_features=512,
                                out_features=256,
                                qi=False,
                                qo=True,
                                num_bits=num_bits[2],
                                )
        self.relu_2 = QReLU(qi=True, num_bits=num_bits[3])

        self.linear_3 = QLinear(in_features=256,
                                out_features=out_channels,
                                qi=False,
                                qo=True,
                                num_bits=num_bits[4],
                                )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_1.simulated_quantize_forward(x).relu()
        x = self.linear_2.simulated_quantize_forward(x).relu()
        x = self.linear_3.simulated_quantize_forward(x)
        return x


def train(model, device, train_loader, optimizer, bit_width_lambda, pbar):
    train_loss = 0
    model.train()
    loss_function = CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        classification_loss = loss_function(output, target)

        desc = (f"FP32 Training "
                f"Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset)}] "
                f"Class. Loss: {classification_loss.item():.4f}")

        if bit_width_lambda is not None:
            bit_width_loss = bit_width_lambda * model.calculate_loss(data)
            desc += f" BitWidth.Loss: {bit_width_loss.item():.4f}"
        else:
            bit_width_loss = tensor([0.0], device=device)

        (classification_loss + bit_width_loss).backward()
        optimizer.step()
        desc += f" BitWidth.Loss: {bit_width_loss.item():.4f}"

        pbar.set_description(desc)
        train_loss += classification_loss.item()

    return train_loss / len(train_loader.dataset)



def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_function = CrossEntropyLoss(reduction="sum")
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += loss_function(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    return test_loss, 100.0 * correct / len(test_loader.dataset)


def validate_simulated_quantize(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_function = CrossEntropyLoss(reduction="sum")
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model.simulated_quantize_forward(data)
        test_loss += loss_function(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    return test_loss, 100.0 * correct / len(test_loader.dataset)


if __name__ == '__main__':
    bit_width = [2, 4, 8]
    batch_size = 512
    epochs = 10
    lr = 0.001
    bit_width_loss = 1e-16

    device = device("cuda" if cuda.is_available() else "cpu")

    train_loader = DataLoader(MNIST("./data",
                                    train=True,
                                    download=True,
                                    transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True,
                              )

    test_loader = DataLoader(MNIST("./data",
                                   train=False,
                                   transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])),
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1,
                             pin_memory=True,
                             )
    x_0, y_0 = next(iter(train_loader))
    num_channels = prod([*x_0.shape[1:]])
    out_channels = y_0.max().item() + 1

    model = MQNetwork(num_channels=num_channels, out_channels=out_channels, num_bits=bit_width).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_loss, best_state_dict = float("inf"), None
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        loss = train(model, device, train_loader, optimizer, bit_width_loss, pbar)
        if loss < best_loss:
            best_loss = loss
            best_state_dict = deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    test_loss, test_accuracy = validate(model, device, test_loader)
    print("Test set: supernet Loss: {:.4f}, Accuracy: {:.4f}%".format(test_loss, test_accuracy))

    print("linear 1", model.linear_1.select_top_k_winners(1))
    print("relu 1", model.relu_1.select_top_k_winners(1))
    print("linear 2", model.linear_2.select_top_k_winners(1))
    print("relu 2", model.relu_2.select_top_k_winners(1))
    print("linear 3", model.linear_3.select_top_k_winners(1))

    wining_bit_width = [sum(model.linear_1.select_top_k_winners(1).values(), []),
                        sum(model.relu_1.select_top_k_winners(1).values(), []),
                        sum(model.linear_2.select_top_k_winners(1).values(), []),
                        sum(model.relu_2.select_top_k_winners(1).values(), []),
                        sum(model.linear_3.select_top_k_winners(1).values(), [])]

    print(f"Integer Operations: {model.estimated_bit_operation_precision(x_0.to(device))}")

    model = QNetwork(num_channels=num_channels, out_channels=out_channels, num_bits=wining_bit_width).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_loss, best_state_dict = float("inf"), None
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        loss = train(model, device, train_loader, optimizer, None, pbar)
        if loss < best_loss:
            best_loss = loss
            best_state_dict = deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    test_loss, test_accuracy = validate(model, device, test_loader)
    print("Test set: quantized Model Loss: {:.4f}, Accuracy: {:.4f}%".format(test_loss, test_accuracy))
