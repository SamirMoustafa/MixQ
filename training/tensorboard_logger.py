import datetime
import warnings
from abc import ABC
from os import listdir, path
from os.path import join
from pathlib import Path

from torch import any
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

DEFAULT_GRAD_NAME = "gradient"
DEFAULT_PARAMETER_NAME = "parameters"


class TensorboardLogger(ABC):
    def __init__(self, log_path: Path | str):
        directories = [d for d in listdir(log_path) if path.isdir(path.join(log_path, d))]
        unique_directory_name = "{:03d}".format(len(directories) + 1)

        self.tensorboard_writer = SummaryWriter(log_dir=join(log_path, unique_directory_name))
        self.current_epoch = 0
        self.is_gradients_extracted = {self.current_epoch: False}

    def __logging_learnable_parameters__(self, module: Module):
        learnable_scaler_param = {}
        # iterating through all parameters
        for name, params in module.named_parameters():
            learnable_scaler_param.update({name: params})
        return learnable_scaler_param

    def __plot_values__(self, tag_name: str, values_as_dictionary: dict):
        for param_name in values_as_dictionary:
            param_value = values_as_dictionary[param_name]
            if any(param_value.isnan()):
                if DEFAULT_GRAD_NAME not in tag_name:
                    warnings.warn("{} has NaN gradiant at epoch {}.".format(param_name, self.current_epoch))
                continue
            tag_name = DEFAULT_PARAMETER_NAME + ("_" + DEFAULT_GRAD_NAME if DEFAULT_GRAD_NAME in param_name else "")
            if param_value.shape.numel() == 1:
                self.tensorboard_writer.add_scalar(
                    tag_name + "/" + param_name, values_as_dictionary[param_name], self.current_epoch
                )
            else:
                if param_value.numel() == 0:
                    warnings.warn("{} has no values.".format(param_name))
                    continue
                self.tensorboard_writer.add_histogram(tag_name + "/" + param_name, param_value, self.current_epoch)

    def __plot_values_gradient__(self, tag_name: str, values_as_dictionary: dict):
        for name, param in values_as_dictionary:
            if param.grad is None:
                if DEFAULT_GRAD_NAME not in tag_name:
                    warnings.warn("{} has no gradiant.".format(name))
                continue
            if any(param.grad.isnan()):
                warnings.warn("{} has NaN gradiant at epoch {}.".format(name, self.current_epoch))
                continue
            if param.shape.numel() == 1:
                self.tensorboard_writer.add_scalar(tag_name + "/" + name, param.grad, self.current_epoch)
            else:
                self.tensorboard_writer.add_histogram(tag_name + "/" + name, param.grad, self.current_epoch)

    def __on_after_backward__(self, module: Module):
        if self.current_epoch in self.is_gradients_extracted and self.is_gradients_extracted[self.current_epoch]:
            return
        self.__plot_values_gradient__(DEFAULT_PARAMETER_NAME + "_" + DEFAULT_GRAD_NAME, module.named_parameters())
        # set `is_gradients_extracted` to be true for current epoch
        self.is_gradients_extracted.update({self.current_epoch: True})

    def training_epoch_end(self, module: Module, epoch: int, training_dict: dict = {}, validation_dict: dict = {}):
        self.current_epoch = epoch

        for tag_sub_name, value in training_dict.items():
            self.tensorboard_writer.add_scalar("{}/training".format(tag_sub_name), value, self.current_epoch)

        for tag_sub_name, value in validation_dict.items():
            self.tensorboard_writer.add_scalar("{}/validation".format(tag_sub_name), value, self.current_epoch)

        learnable_scaler_param = self.__logging_learnable_parameters__(module)
        self.__plot_values__(DEFAULT_PARAMETER_NAME, learnable_scaler_param)
        self.__on_after_backward__(module)
