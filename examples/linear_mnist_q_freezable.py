from math import prod
from tqdm import tqdm

from torch.optim import SGD
from torch import cuda, device
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from quantization.fixed_modules.parametric.linear import QLinear, QLinearBatchNormReLU


class QNetwork(Module):
    def __init__(self, num_channels, out_channels, num_bits):
        super(QNetwork, self).__init__()
        self.linear_bn_relu_1 = QLinearBatchNormReLU(in_features=num_channels,
                                                     out_features=512,
                                                     qi=True,
                                                     qo=True,
                                                     num_bits=num_bits[0],
                                                     )
        self.linear_bn_relu_2 = QLinearBatchNormReLU(in_features=512,
                                                     out_features=256,
                                                     qi=False,
                                                     qo=True,
                                                     num_bits=num_bits[1],
                                                     )

        self.fc_1 = QLinear(in_features=256,
                            out_features=out_channels,
                            qi=False,
                            qo=True,
                            num_bits=num_bits[2],
                            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_bn_relu_1(x)
        x = self.linear_bn_relu_2(x)
        x = self.fc_1(x)
        return x

    def simulated_quantize_forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_bn_relu_1.simulated_quantize_forward(x)
        x = self.linear_bn_relu_2.simulated_quantize_forward(x)
        x = self.fc_1.simulated_quantize_forward(x)
        return x

    def freeze(self):
        self.linear_bn_relu_1.freeze()
        self.linear_bn_relu_2.freeze(qi=self.linear_bn_relu_1.qo)
        self.fc_1.freeze(qi=self.linear_bn_relu_2.qo)

    def quantize_inference(self, x):
        x = x.view(x.size(0), -1)
        qx = self.linear_bn_relu_1.qi.quantize(x)
        qx = self.linear_bn_relu_1.quantize_inference(qx)
        qx = self.linear_bn_relu_2.quantize_inference(qx)
        qx = self.fc_1.quantize_inference(qx)
        x = self.fc_1.qo.dequantize(qx)
        return x


def train(model, device, train_loader, optimizer, pbar):
    model.train()
    loss_function = CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"FP32 Training Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")


def quantize_aware_training(model, device, train_loader, optimizer, pbar):
    train_loss = 0
    model.train()
    loss_function = CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.simulated_quantize_forward(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_description(f"Quantize Aware Training Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
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


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return 100.0 * correct / len(test_loader.dataset)


if __name__ == '__main__':
    bit_width = [[8, 8, 8],
                 [-1, 8, 8],
                 [-1, 8, 8],
                 ]
    batch_size = 64
    epochs = 10
    lr = 0.01
    lr_quant = 0.0001
    momentum = 0.5

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
    model = QNetwork(num_channels=num_channels, out_channels=out_channels, num_bits=bit_width).to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train(model, device, train_loader, optimizer, pbar)
    test_loss, test_accuracy = validate(model, device, test_loader)
    print("Test set: FP32 Model Loss: {:.4f}, Accuracy: {:.4f}%".format(test_loss, test_accuracy))

    optimizer = SGD(model.parameters(), lr=lr_quant, momentum=momentum)
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        quantize_aware_training(model, device, train_loader, optimizer, pbar)

    test_loss, test_accuracy = validate_simulated_quantize(model, device, test_loader)
    print("Test set: Simulated Quantized Model Loss: {:.4f}, Accuracy: {:.4f}%".format(test_loss, test_accuracy))

    model.eval()
    model.freeze()
    test_accuracy = quantize_inference(model, test_loader)
    print("Test set: Fully Quantized Model Accuracy: {:.4f}%\n".format(test_accuracy))
