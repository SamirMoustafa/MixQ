from copy import deepcopy
from os import mkdir
from os.path import exists

from tqdm import tqdm
from torch import no_grad
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import StepLR

from training.tensorboard_logger import TensorboardLogger
from quantization.utility import camel_case_split


def calibrate(model, data):
    model.train()
    x, edge_index = data.x, data.edge_index
    return model.calibrate(x, edge_index)


def train(model, optimizer, scheduler, data):
    model.train()
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
    optimizer.zero_grad()
    out = model(x, edge_index, edge_weight)
    loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss)



@no_grad()
def validate(model, data, metrics):
    model.eval()
    x, edge_index = data.x, data.edge_index
    y_pred = model(x, edge_index, data.edge_weight)
    metrics = [metric.to(x.device) for metric in metrics]
    metrics_values = []
    for mask in [data.train_mask, data.val_mask]:
        metrics_values.append([metric(y_pred[mask], data.y[mask]) for metric in metrics])
    return metrics_values


def train_evaluate_model(epochs, model, optimizer, data, metrics, scheduler=None, log_directory="./tensorboard"):
    if log_directory is not None and not exists(log_directory):
        mkdir(log_directory)
    if log_directory is not None:
        tensorboard_logger = TensorboardLogger(log_directory)
    if scheduler is None:
        scheduler = StepLR(optimizer, step_size=1, gamma=1)
    best_epoch, best_metric_0, best_state_dict = 0, 0, deepcopy(model.state_dict())
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        loss = train(model, optimizer, scheduler, data)
        training_metrics_values, validate_metrics_values = validate(model, data, metrics)
        pbar.set_description(f"Train loss: {loss:.2f} | Best validation {camel_case_split(metrics[0].__class__.__name__)} {best_metric_0:.3f} | epoch {best_epoch}")
        training_dict = {"loss": loss,
                         "learning rate": optimizer.param_groups[0]["lr"],
                         }
        training_dict.update(
            {
                metric.__class__.__name__: training_metric_values
                for metric, training_metric_values in zip(*[metrics, training_metrics_values])
            }
        )
        validation_dict = {
            metric.__class__.__name__: training_metric_values
            for metric, training_metric_values in zip(*[metrics, validate_metrics_values])
        }
        if log_directory is not None:
            tensorboard_logger.training_epoch_end(model, epoch, training_dict, validation_dict)
        if validate_metrics_values[0] > best_metric_0:
            best_epoch, best_metric_0 = epoch, validate_metrics_values[0]
            best_state_dict = deepcopy(model.state_dict())
    return best_epoch, best_metric_0, best_state_dict
