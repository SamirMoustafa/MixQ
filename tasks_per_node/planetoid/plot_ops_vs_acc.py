import time
import pickle
from io import BytesIO
from os.path import abspath, exists, dirname, realpath, join

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt

import plotly
import plotly.express as px
import plotly.graph_objects as go

from torch import cuda, device, no_grad
from torch.nn import ReLU
from torch.optim import AdamW
from torch.nn.functional import cross_entropy

from torchmetrics import Accuracy

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Sequential, MLP, GCNConv, GATConv, GINConv, SAGEConv, TransformerConv, TAGConv, SuperGATConv

from torch_operation_counter import OperationsCounterMode


def define_model(layer_class: type,
                 layer_arguments: dict,
                 hidden_channels: int,
                 number_of_layers: int,
                 add_self_loops: bool = True,
                 ) -> Sequential:
    layers_args = [layer_arguments.copy() for _ in range(number_of_layers)]
    if number_of_layers >= 2:
        layers_args[0]["out_channels"] = hidden_channels
        for i in range(number_of_layers - 1):
            layers_args[i + 1]["in_channels"] = hidden_channels
            if i + 1 < number_of_layers - 1:
                layers_args[i + 1]["out_channels"] = hidden_channels
    if layer_class in [GINConv, ]:
        layers_args = [{"nn": MLP([layer_arguments['in_channels'], layer_arguments['out_channels']])}
                       for layer_arguments in layers_args]
    if layer_class in [GCNConv, GATConv, SuperGATConv]:
        [layer_arguments.update({"add_self_loops": add_self_loops}) for layer_arguments in layers_args]

    layers = [layer_class(**layer_arguments) for layer_arguments in layers_args]
    model_list = []
    for i, layer_i in enumerate(layers):
        internal_layer_arguments_str = f"x, edge_index -> x{i}" if i <= 0 else f"x{i - 1}, edge_index -> x{i}"
        if i == number_of_layers - 1:
            model_list.extend(((layer_i, internal_layer_arguments_str),))
        else:
            model_list.extend(((layer_i, internal_layer_arguments_str), ReLU(inplace=True)))
    return Sequential("x, edge_index", model_list)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    log_probs = model(data.x, data.edge_index)
    loss = cross_entropy(log_probs[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@no_grad()
def validate(model, data, metric):
    model.eval()
    log_probs = model(data.x, data.edge_index)
    return metric(log_probs[data.val_mask].argmax(dim=1), data.y[data.val_mask]).item()


def train_and_evaluate_gnn(model, data, lr, num_class=None):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_class = data.y.unique().shape[0] if num_class is None else num_class
    accuracy = Accuracy(task="multiclass", num_classes=num_class).to(data.x)
    best_epoch, best_acc = 0, 0
    for epoch in range(epochs):
        loss = train(model, optimizer, data)
        acc = validate(model, data, accuracy)
        best_epoch, best_acc = (epoch, acc) if acc > best_acc else (best_epoch, best_acc)
    return best_epoch, 100 * best_acc


def construct_and_evaluate_model(number_of_runs, layer_cls, layer_args, hidden_channels, num_layers, data, lr,
                                 num_class):
    epochs_list, acc_list = [], []
    for _ in range(number_of_runs):
        model = define_model(layer_cls, layer_args, hidden_channels, num_layers)
        assert model(x, edge_index).shape[0] == y.shape[0], f"Problem with {model}"
        model = model.to(device)
        data = data.to(device)
        best_epoch, best_acc = train_and_evaluate_gnn(model, data, lr, num_class)
        epochs_list += [best_epoch]
        acc_list += [best_acc]
    return np.mean(epochs_list), np.mean(acc_list)


if __name__ == '__main__':
    color_sequence = px.colors.qualitative.Plotly

    device = device("cuda" if cuda.is_available() else "cpu")
    dataset_name = "Cora"
    num_of_runs = 5
    epochs = 300

    layers_cls = [GCNConv,
                  GATConv,
                  GINConv,
                  TransformerConv,
                  TAGConv,
                  SuperGATConv]

    num_of_layers = [1, 2, 3, 4, 5]
    lr_options = [1e-1, 1e-2, 1e-3]
    hidden_channels_options = [32, 64, 128]
    layers_additional_args_options = {GCNConv: {"aggr": ["add", "mean"]},
                                      GATConv: {"negative_slope": [0.1, 0.2, 0.5]},
                                      GINConv: {"train_eps": [True, False]},
                                      TransformerConv: {"root_weight": [True, False]},
                                      TAGConv: {"K": [2, 3, 4]},
                                      SuperGATConv: {"negative_slope": [0.1, 0.2, 0.5]},
                                      }

    number_to_exponential = lambda x: f"{x:.0e}"
    file_name = (f"{dataset_name.lower()}_"
                 f"number_of_layers-{'-'.join([str(num_layer) for num_layer in num_of_layers])}_"
                 f"epochs-{epochs}_"
                 f"lrs-{'-'.join([number_to_exponential(lr) for lr in lr_options])}_"
                 f"hidden_channels-{'-'.join([str(hidden_channel) for hidden_channel in hidden_channels_options])}"
                 f".pylst")

    dataset_dict = {}
    datasets_statistics = {}
    path = join(dirname(realpath(abspath(""))), '../data', dataset_name)
    data = Planetoid(path, dataset_name)[0]
    x, edge_index, y = data.x, data.edge_index, data.y
    num_features, num_class = data.num_features, len(y.unique())
    dataset_dict[dataset_name] = data
    datasets_statistics[dataset_name] = {"num_features": num_features,
                                         "num_class": num_class}

    df_columns = ["dataset_name",
                  "layer",
                  "num_of_layers",
                  "operations",
                  "size_MB",
                  "epochs",
                  "inference",
                  "accuracy",
                  "hyperparameters"]
    df_rows = []

    if exists(file_name):
        # If the file exists, load the list from the file
        with open(file_name, "rb") as file:
            df_rows = pickle.load(file)
    else:
        # If the file doesn't exist, perform some code and save the list to the file
        data = dataset_dict[dataset_name].to(device)
        num_class = datasets_statistics[dataset_name]["num_class"]

        pbar = tqdm(layers_cls)
        for layer_cls in pbar:
            for num_layers in num_of_layers:
                best_hyperparameters = {"lr": None, "hidden_channels": None, "layer_args": None}
                best_epoch, best_acc = 0, 0
                for lr in lr_options:
                    for hidden_channels in hidden_channels_options:
                        for layer_args_key, layer_args_values in layers_additional_args_options[layer_cls].items():
                            for layer_args_value in layer_args_values:
                                in_channels = datasets_statistics[dataset_name]["num_features"]
                                out_channels = datasets_statistics[dataset_name]["num_class"]
                                layer_args = {"in_channels": in_channels, "out_channels": out_channels}
                                layer_args[layer_args_key] = layer_args_value
                                epoch, acc = construct_and_evaluate_model(num_of_runs,
                                                                          layer_cls,
                                                                          layer_args,
                                                                          hidden_channels,
                                                                          num_layers,
                                                                          data,
                                                                          lr,
                                                                          num_class)
                                if acc > best_acc:
                                    best_hyperparameters = {"lr": lr,
                                                            "hidden_channels": hidden_channels,
                                                            "layer_args": layer_args}
                                    best_epoch, best_acc = epoch, acc

                layer_name = layer_cls.__name__
                model = define_model(layer_cls,
                                     best_hyperparameters["layer_args"],
                                     best_hyperparameters["hidden_channels"],
                                     num_layers,
                                     False
                                     ).to(device)
                ops_counter = OperationsCounterMode(model)
                model(data.x, data.edge_index)
                with ops_counter:
                    model(data.x, data.edge_index)
                start = time.time()
                model(data.x, data.edge_index)
                end = time.time()
                elapsed_time = end - start
                model_size_MB = sum([p.numel() for p in [*model.parameters()]]) * 4 / (1024 ** 2)
                df_rows.append([dataset_name,
                                layer_name,
                                num_layers,
                                ops_counter.total_operations,
                                model_size_MB,
                                best_epoch,
                                elapsed_time,
                                best_acc,
                                best_hyperparameters])
                pbar.set_description(f"Processed {layer_name} "
                                     f"{num_layers} layers "
                                     f"lr {best_hyperparameters['lr']} "
                                     f"hidden_channels {best_hyperparameters['hidden_channels']} "
                                     f"with {best_acc:.2f}% accuracy")
        with open(file_name, "wb") as file:
            pickle.dump(df_rows, file)

    df = pd.DataFrame(df_rows, columns=df_columns)

    x_axis, y_axis, z_axis = "operations", "accuracy", "size_MB"
    color, size = "layer", "size_MB"
    labels = {
        "operations": "<b>Number of Operations</b>",
        "accuracy": f"<b>Accuracy (%)</b>",
        "layer": "<b>Layer Type</b>"
    }

    x_values = df[x_axis].values
    y_values = df[y_axis].values
    z_values = df[z_axis].values

    corr_between_x_y, p_value_between_x_y = stats.spearmanr(x_values, y_values)
    corr_between_z_y, p_value_between_z_y = stats.spearmanr(z_values, y_values)

    print(f"Correlation between {x_axis} and {y_axis} is {corr_between_x_y:.3f} with p-value {p_value_between_x_y:.1e}")
    print(f"Correlation between {z_axis} and {y_axis} is {corr_between_z_y:.3f} with p-value {p_value_between_z_y:.1e}")

    df["size_MB"] = df["size_MB"].clip(df["size_MB"].mode()[2], max(df["size_MB"]))
    df["layer"] = df["layer"].apply(lambda x: x.replace("Conv", ""))

    title = f"<b>{labels[x_axis]} vs {labels[y_axis]} with Correlation {corr_between_x_y:.2f}</b>"

    fig = px.scatter(df,
                     x=x_axis,
                     y=y_axis,
                     color=color,
                     size=size,
                     template="plotly_white",
                     labels=labels,
                     log_x=False,
                     log_y=True,
                     width=500,
                     height=300,
                     color_discrete_sequence=px.colors.qualitative.Dark24_r
                     )

    tick_vals = pd.to_numeric([f"{n:.1g}" for n in np.linspace(1024, df[x_axis].max(), 6)])
    tick_text = [
        f"{val / 1e3:.0f} Kilo" if val < 1e6 else
        f"{val / 1e6:.0f} Mega" if val < 1e9 else
        f"{val / 1e9:.0f} Giga" for val in tick_vals
    ]

    fig.update_xaxes(showline=True,
                     linewidth=1,
                     linecolor="black",
                     tickmode="array",
                     tickvals=tick_vals,
                     ticktext=tick_text,
                     showgrid=True,
                     zeroline=True,
                     showticklabels=True,
                     gridwidth=0,
                     ticks="outside",
                     tickson="boundaries",
                     ticklen=10,
                     tickfont_family="Serif",
                     )
    fig.update_yaxes(showline=True,
                     linewidth=1,
                     linecolor="black",
                     showgrid=True,
                     zeroline=True,
                     showticklabels=True,
                     gridwidth=1,
                     minor_griddash="dot",
                     ticks="outside",
                     tickson="boundaries",
                     ticklen=10,
                     tickfont_family="Serif",
                     )

    fig.update_layout(title_x=0.5, font_family="Serif", )
    fig.update_traces(marker=dict(line=dict(width=0.1, color='black')))

    dummy_sizes = [10, 20, 30]
    dummy_labels = ['0.5 MB', '1.5 MB', '3.5 MB']

    quartile_1 = df[x_axis].quantile(0.82)
    quartile_2 = df[x_axis].quantile(0.99)
    max_y = df[y_axis].min() * 1.05

    dummy_data = pd.DataFrame({
        'x': np.linspace(quartile_1, quartile_2, len(dummy_sizes)),
        'y': [df[y_axis].min() * 1.01] * len(dummy_sizes),
        'size': dummy_sizes,
        'label': dummy_labels,
    })

    fig.add_shape(
        type="rect",
        x0=quartile_1 * 0.85, y0=max_y * 0.945, x1=quartile_2 * 1.1, y1=max_y,
        line=dict(color="white"),
        fillcolor="white",
        opacity=0.9,
        layer="below"
    )

    for i, row in dummy_data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['x']], y=[row['y']], mode="markers+text",
                text=[f"<b>{row['label']}</b>"], textposition="top center",
                marker=dict(size=row['size'], color="#d3d3d3", line=dict(color='darkblue')),
                textfont=dict(size=10),
                showlegend=False,
            )
        )

    plotly.io.write_html(fig, f"{dataset_name.lower()}_{x_axis}_vs_{y_axis}_corr_{corr_between_x_y:.2f}.html")

    # Display the plot
    png_buffer = BytesIO()
    fig.write_image(png_buffer, format="png")
    png_buffer.seek(0)
    graph_image = Image.open(png_buffer)
    image_width, image_height = graph_image.size
    desired_dpi = 100
    fig_width = image_width / desired_dpi
    fig_height = image_height / desired_dpi
    plt.figure(figsize=(fig_width, fig_height), dpi=desired_dpi)
    plt.imshow(graph_image, aspect='auto')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
