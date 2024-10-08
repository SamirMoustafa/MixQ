# This script is based on this implementation of GraphSAGE on Reddit dataset:
# https://github.com/atanuroy911/graphsage-reddit/

from torch import no_grad, cat
from torch.nn import Module, ModuleList
from torch.nn.functional import dropout, cross_entropy
from tqdm import tqdm

from fixed_graph_sage import QGraphSAGE


class SAGE(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = ModuleList()
        q_temp = [32, 32, 32, 32, 32, 32, 32, 32]
        self.convs.append(QGraphSAGE(in_channels, hidden_channels, q_temp, q_temp, q_temp))
        self.convs.append(QGraphSAGE(hidden_channels, out_channels, q_temp, q_temp, q_temp))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = dropout(x, p=0.5, training=self.training)
        return x

    @no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description("Evaluating")
        device = x_all.device
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                x = conv(x.to(device), batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].detach().cpu())
                pbar.update(batch.batch_size)
            x_all = cat(xs, dim=0)
        pbar.close()
        return x_all

    def estimated_bit_operation_precision(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description("Estimating Bit Operation")
        bit_operation_per_layer = []
        device = x_all.device
        for i, conv in enumerate(self.convs):
            xs = []
            bit_operation_per_batch = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                bit_operation = conv.estimated_bit_operation_precision(x.to(device), batch.edge_index.to(device))
                bit_operation_per_batch += [bit_operation]
                x = conv(x.to(device), batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].detach().cpu())
                pbar.update(batch.batch_size)
            bit_operation_per_layer += [np.mean(bit_operation_per_batch)]
            x_all = cat(xs, dim=0)
        pbar.close()
        return np.sum(bit_operation_per_layer)



def train(epoch, model, train_loader, optimizer):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()
    loss = total_loss / total_examples
    accuracy = total_correct / total_examples * 100
    return loss, accuracy

@no_grad()
def evaluate(model, data, subgraph_loader):
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    pred = y_hat.to(data.y.device)
    accuracies = [(pred[mask] == data.y[mask]).float().mean().item() * 100
                  for mask in (data.train_mask, data.val_mask, data.test_mask)]
    return accuracies


if __name__ == '__main__':
    import copy
    import argparse

    import numpy as np
    from torch import cuda, device as torch_device
    from torch.optim import NAdam
    from torch_geometric.datasets import Reddit
    from torch_geometric.loader import NeighborLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    arguments = vars(args)
    [print(f"{k}: {v}") for k, v in arguments.items()]

    device = torch_device("cuda" if cuda.is_available() else "cpu")

    dataset = Reddit("../../data/reddit/")
    num_nodes = dataset[0].num_nodes
    num_edges = dataset[0].num_edges

    data = dataset[0].to(device, "x", "y")
    kwargs = {"batch_size": args.batch_size, "num_workers": 6, "persistent_workers": True}
    train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[25, 10], shuffle=True, **kwargs)
    subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None, num_neighbors=[-1], shuffle=False, **kwargs)

    val_accuracies, test_accuracies = [], []
    model = SAGE(dataset.num_features, args.hidden_channels, dataset.num_classes).to(device)

    for run_i in range(args.num_runs):
        best_val_accuracy, best_epoch, test_accuracy = -float("inf"), -1, -float("inf")

        model.reset_parameters()
        optimizer = NAdam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            loss, accuracy = train(epoch, model, train_loader, optimizer)
            train_accuracy, val_accuracy, test_accuracy = evaluate(model, data, subgraph_loader)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                test_accuracy = test_accuracy

        val_accuracies.append(best_val_accuracy)
        test_accuracies.append(test_accuracy)
        print(f"Run {run_i + 1:02d}, Best Validation Accuracy: {best_val_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

    bit_operation = model.estimated_bit_operation_precision(data.x, subgraph_loader)
    print(f"Bit Operation: {bit_operation:.2f}")

    print(f"Validation Accuracies: {np.mean(val_accuracies):.2f} ± {np.std(val_accuracies):.2f}")
    print(f"Test Accuracies: {np.mean(test_accuracies):.2f} ± {np.std(test_accuracies):.2f}")
