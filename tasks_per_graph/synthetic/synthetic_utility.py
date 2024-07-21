import numpy as np
import scipy as sp
from tqdm import tqdm

from torch import float32, from_numpy, no_grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def positional_encoding(edge_index, pos_enc_dim=50):
    number_of_nodes = edge_index.max().item() + 1
    # Laplacian
    A = sp.sparse.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(number_of_nodes, number_of_nodes))
    N = sp.sparse.diags(degree(edge_index[0], dtype=float32).clip(1).numpy() ** -0.5, dtype=float)
    L = sp.eye(number_of_nodes) - N * A * N
    # Eigenvectors with scipy
    EigVal, EigVec = sp.sparse.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    positional_encode = from_numpy(np.real(EigVec[:, 1:pos_enc_dim + 1])).float()
    return positional_encode


def train_step(model, data, criterion, optimizer, bit_width_lambda=None):
    output = model(data)
    loss_train = criterion(output, data.y)
    if bit_width_lambda is not None:
        loss_train = loss_train + bit_width_lambda * model.calculate_weighted_loss(data)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    return output, loss_train


@no_grad()
def eval_step(model, data, criterion):
    output = model(data)
    loss_test = criterion(output, data.y)
    return output, loss_test


def train_iteration(model, loader, criterion, optimizer, device, bit_width_lambda=None):
    model.train()
    total_loss = 0
    correct_count = 0
    for data in loader:
        data = data.to(device)
        output, loss = train_step(model, data, criterion, optimizer, bit_width_lambda)
        total_loss += loss.item() * data.num_graphs
        preds = output.argmax(dim=1).type_as(data.y)
        correct_count += preds.eq(data.y).sum().item()
    return total_loss / len(loader.dataset), correct_count / len(loader.dataset) * 100


def evaluate_iteration(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_count = 0
    for data in loader:
        data = data.to(device)
        output, loss = eval_step(model, data, criterion)
        total_loss += loss.item() * data.num_graphs
        preds = output.argmax(dim=1).type_as(data.y)
        correct_count += preds.eq(data.y).sum().item()
    return total_loss / len(loader.dataset), correct_count / len(loader.dataset) * 100


def classification_task(model, loaders, device, epochs=200, lr=1e-2, bit_width_lambda=None):
    train_loader, val_loader, test_loader = loaders
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    criterion = CrossEntropyLoss()

    best_val_loss, best_model_state_dict = float("inf"), None
    pbar_train = tqdm(range(epochs), desc="Training Progress")
    for epoch in pbar_train:
        train_loss, train_acc = train_iteration(model, train_loader, criterion, optimizer, device, bit_width_lambda)
        val_loss, val_acc = evaluate_iteration(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        pbar_train.set_description(f"Epoch {epoch + 1} Train Loss {train_loss:.3f} Train Acc {train_acc:.1f} Val Loss {val_loss:.3f} Val Acc {val_acc:.1f}")

        if val_loss < best_val_loss:
            best_val_loss = val_acc
            best_model_state_dict = model.state_dict().copy()

    model.load_state_dict(best_model_state_dict)
    train_acc = evaluate_iteration(model, train_loader, criterion, device)[1]
    val_acc = evaluate_iteration(model, val_loader, criterion, device)[1]
    test_acc = evaluate_iteration(model, test_loader, criterion, device)[1]
    return train_acc, val_acc, test_acc

