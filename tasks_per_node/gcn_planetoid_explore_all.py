import argparse
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from itertools import product
from os import makedirs
from os.path import exists
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame, concat
from plotly.graph_objs import Scatter, Figure
from plotly.io import write_html
from tqdm import tqdm

from torch.nn import Module
from torch.optim import Adam
from torch import cuda, device, no_grad
from torch.nn.functional import cross_entropy, dropout

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from training.logger import setup_logger
from quantization.fixed_modules.non_parametric import QReLU
from quantization.fixed_modules.parametric import QGraphConvolution
from utility import format_fraction, flatten_tuple


class QGCN(Module):
    def __init__(self, num_channels: int, hidden_channels: int, out_channels: int, num_bits: List[int]):
        super(QGCN, self).__init__()
        self.gcn_1 = QGraphConvolution(in_channels=num_channels,
                                       out_channels=hidden_channels,
                                       qi=True,
                                       qo=True,
                                       num_bits=num_bits[0],
                                       is_signed=False,
                                       quantize_per="column",
                                       )
        self.relu_1 = QReLU(num_bits=num_bits[1], quantize_per="column")
        self.gcn_2 = QGraphConvolution(in_channels=hidden_channels,
                                       out_channels=out_channels,
                                       qi=False,
                                       qo=True,
                                       num_bits=num_bits[2],
                                       quantize_per="column",
                                       )
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
        relu_integer_operations = self.relu_1.estimated_bit_operation_precision(x)
        x = self.relu_1(x)
        gcn_2_operations = self.gcn_2.estimated_bit_operation_precision(x, edge_index, edge_attr)
        return gcn_1_operations + relu_integer_operations + gcn_2_operations


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    classification_loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
    classification_loss.backward()
    optimizer.step()
    return classification_loss.item()


@no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)
    accuracies = [(pred[mask] == data.y[mask]).float().mean().item() * 100
                  for mask in (data.train_mask, data.val_mask, data.test_mask)]
    return accuracies


def training_loop(epochs, model, optimizer, data, verbose=False):
    best_val_accuracy, best_epoch, test_accuracy = -float("inf"), 0, 0
    best_state_dict = model.state_dict()
    pbar = tqdm(range(1, epochs + 1)) if verbose else range(1, epochs + 1)
    for epoch in pbar:
        loss = train(model, optimizer, data)
        train_accuracy, validation_accuracy, tmp_test_accuracy = evaluate(model, data)
        if validation_accuracy > best_val_accuracy:
            best_epoch = epoch
            best_val_accuracy = validation_accuracy
            test_accuracy = tmp_test_accuracy
            best_state_dict = deepcopy(model.state_dict().copy())
        if verbose:
            pbar.set_description(f"{epoch:03d}/{epochs:03d},"
                                 f"Loss:{loss:.2f},TrainAcc:{train_accuracy:.2f},"
                                 f"ValAcc:{validation_accuracy:.2f},"
                                 f"BestValAcc:{best_val_accuracy:.2f},"
                                 f"BestEpoch:{best_epoch:03d}")
    return best_state_dict, test_accuracy


def pareto_front(Xs, Ys):
    sorted_points = sorted(zip(Xs, Ys), key=lambda x: x[1], reverse=True)
    pareto_front = [sorted_points[0]]
    for point in sorted_points[1:]:
        if point[0] < pareto_front[-1][0]:
            pareto_front.append(point)
    return zip(*pareto_front)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="Cora")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_quant", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="explore_all_logs")

    parser.add_argument("--num_bits_list", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()

    arguments = vars(args)

    log_file_name = (f"{args.dataset_name}/"
                     f"hidden_{args.hidden_channels}/"
                     f"wd_{format_fraction(args.weight_decay)}/"
                     f"lr_{format_fraction(args.lr)}/"
                     f"bit_width_{','.join(map(str, args.num_bits_list))}"
                     )
    if not exists(args.log_dir + "/" + log_file_name):
        makedirs(args.log_dir + "/" + log_file_name)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name += f"/log_{current_time}"
    logger = setup_logger(filename=f"{args.log_dir}/{log_file_name}.log", verbose=False)

    [logger.info(f"{k}: {v}") for k, v in arguments.items()]
    logger.info("=" * 150)

    device = device(args.device)

    dataset = Planetoid(root="../data", name=args.dataset_name, transform=NormalizeFeatures())
    data = dataset[0].to(device)

    layer_names_per_index = ["$X$",
                             "$X \Theta^{(0)}$",
                             "$\Theta^{(0)}$",
                             "$\hat{A}^{(0)}$",
                             "$\hat{A}^{(0)} X \Theta^{(0)}$",
                             "$\sigma(\hat{A}^{(0)} X \Theta^{(0)}) \Theta^{(1)}$",
                             "$\Theta^{(1)}$",
                             "$\hat{A}^{(1)}$",
                             "$\hat{A}^{(1)} \sigma(\hat{A}^{(0)} X \Theta^{(0)}) \Theta^{(1)}$"]
    bit_width_structures = [["?", "?", "?", "?", None, "?"], [None], [None, "?", "?", "?", None, "?"]]
    bit_width_combinations = [list(product(*([args.num_bits_list if x == "?" else [x] for x in sub])))
                              for sub in bit_width_structures]

    df = DataFrame(columns=["Bit Widths", "Accuracy", "Std. Dev.", "Average Bit Width"])
    fp32_accuracies = []
    pbar = tqdm(product(*bit_width_combinations), total=len(args.num_bits_list) ** 9, desc="Bit Widths")
    for bit_width_i in pbar:
        quantize_accuracies = []
        average_bit_width = np.mean([*filter(lambda x: x != None, flatten_tuple(bit_width_i))])
        for run_i in range(args.num_runs):
            # Training Full Precision Model
            model = QGCN(dataset.num_features, args.hidden_channels, dataset.num_classes, bit_width_i).to(device)
            model.set_forward_func(model.full_precision_forward)
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_state_dict, test_accuracy = training_loop(args.epochs, model, optimizer, data)
            fp32_accuracies.append(test_accuracy)
            # Training Quantized Model
            model.load_state_dict(best_state_dict)
            model.set_forward_func(model.simulated_quantize_forward)
            optimizer = Adam(model.parameters(), lr=args.lr_quant, weight_decay=args.weight_decay)
            best_state_dict, test_accuracy = training_loop(args.epochs, model, optimizer, data)
            quantize_accuracies.append(test_accuracy)

        bit_operation_precision = model.estimated_bit_operation_precision(data.x, data.edge_index, data.edge_attr) / 1e9

        logger.info("=" * 150)

        logger.info(f"Bit Widths: {[*filter(lambda x: x != None, flatten_tuple(bit_width_i))]} | "
                    f"Accuracy: {np.mean(quantize_accuracies):.2f}% ± {np.std(quantize_accuracies):.2f}% | "
                    f"Average Bit Width: {average_bit_width:.2f} | "
                    f"Estimated Bit Operation Precision: {bit_operation_precision:.2f} GOp")
        new_row = DataFrame({"Bit Widths": [[*filter(lambda x: x != None, flatten_tuple(bit_width_i))]],
                             "Accuracy": [np.mean(quantize_accuracies)],
                             "Std. Dev.": [np.std(quantize_accuracies)],
                             "Average Bit Width": [average_bit_width],
                             "Estimated Bit Operation Precision": [bit_operation_precision]})
        df = concat([df, new_row], ignore_index=True)

    logger.info(f"FP32 Accuracies Mean: {np.mean(fp32_accuracies):.2f}% ± {np.std(fp32_accuracies):.2f}%")
    logger.info("=" * 150)
    logger.info(df.to_markdown(index=False))

    df.to_csv(f"{args.log_dir}/{log_file_name}_results.csv", index=False)

    pareto_bit_widths, pareto_accuracies = pareto_front(df["Average Bit Width"], df["Accuracy"])

    fig = Figure()

    # All choices
    fig.add_trace(Scatter(
        x=df["Average Bit Width"],
        y=df["Accuracy"],
        mode="markers",
        name="<b>Quantized Architectures</b>",
        marker=dict(color="#0065a7", symbol="circle-open-dot", size=5),
        text=[
            f"Bit Widths: {bit_widths}<br>Accuracy: {accuracy:.2f}%<br>Estimated Bit Operation Precision: {bit_operation_precision:.2f} GOp"
            for bit_widths, accuracy, bit_operation_precision in
            zip(df["Bit Widths"], df["Accuracy"], df["Estimated Bit Operation Precision"])],
    ))

    # FP32 Lines
    fig.add_trace(Scatter(
        x=[df["Average Bit Width"].min() * 0.95, df["Average Bit Width"].max() * 1.05],
        y=[np.mean(fp32_accuracies), ] * 2,
        mode='lines',
        name="<b>FP32 Architecture</b>",
        line=dict(color="#885078", width=2, dash="dash")
    ))
    fig.add_trace(Scatter(
        x=[df["Average Bit Width"].min() * 0.95, df["Average Bit Width"].max() * 1.05],
        y=[np.mean(fp32_accuracies) + np.std(fp32_accuracies), ] * 2,
        mode='lines',
        marker=dict(color="#885078"),
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(Scatter(
        x=[df["Average Bit Width"].min() * 0.95, df["Average Bit Width"].max() * 1.05],
        y=[np.mean(fp32_accuracies) - np.std(fp32_accuracies), ] * 2,
        mode='lines',
        marker=dict(color="#885078"),
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(136, 80, 120, 0.3)',
        showlegend=False
    ))

    # Pareto Frontier
    fig.add_trace(Scatter(
        x=list(pareto_bit_widths),
        y=list(pareto_accuracies),
        mode="lines+markers",
        name="<b>Pareto Front</b>",
        line=dict(color="#c76cc2", width=2),
        marker=dict(symbol="circle", size=5),
    ))
    fig.update_xaxes(showline=True, linewidth=1, griddash="dot", linecolor="black", tickmode="array",
                     ticks="outside", tickson="boundaries", ticklen=10, tickfont_family="Arial Black")
    fig.update_yaxes(showline=True, linewidth=1, griddash="dash", linecolor="black", showgrid=True, ticks="outside",
                     tickson="boundaries", ticklen=10, tickfont_family="Arial Black", gridcolor="lightgray")

    fig.update_layout(
        title_x=0.5,
        xaxis_title="<b>Average Bit Width</b>",
        yaxis_title="<b>Accuracy (%)<b>",
        template="plotly_white",
        legend=dict(orientation="v", yanchor="bottom", xanchor="left", x=0.4, y=0.0, bgcolor="rgba(0,0,0,0)"),
        barmode="stack",
        xaxis={'categoryorder': 'total ascending'},
        margin=dict({'l': 0, 'r': 0, 't': 0, 'b': 0}),
        width=400, height=350,
        font=dict(family="Arial Black", size=12),
    )

    fig.update_traces(marker=dict(line=dict(width=0.1, color='black')))

    write_html(fig, f"{args.log_dir}/{log_file_name}_plot.html")

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

    pareto_bit_widths, pareto_accuracies = list(pareto_bit_widths), list(pareto_accuracies)
    pareto_bit_widths_models = [
        df["Bit Widths"][(df["Average Bit Width"] == average_bit_width_i) & (df["Accuracy"] == accuracy)].values[0]
        for average_bit_width_i, accuracy in zip(pareto_bit_widths, pareto_accuracies)
    ]

    for pareto_bit_widths_i, pareto_accuracy_i in zip(pareto_bit_widths_models, pareto_accuracies):
        logger.info(f"Pareto Front: Bit Widths: {pareto_bit_widths_i} | Accuracy: {pareto_accuracy_i:.2f}%")

    bit_widths_pareto_per_layer = np.stack(pareto_bit_widths_models).T

    pareto_bit_widths, pareto_accuracies = list(pareto_bit_widths), list(pareto_accuracies)
    pareto_bit_widths_models = [
        df["Bit Widths"][(df["Average Bit Width"] == average_bit_width_i) & (df["Accuracy"] == accuracy)].values[0]
        for average_bit_width_i, accuracy in zip(pareto_bit_widths, pareto_accuracies)
    ]

    df_pareto = DataFrame({"Index": range(1, len(layer_names_per_index) + 1),
                           "Layer": layer_names_per_index,
                           "Bit Widths": [bi for bi in bit_widths_pareto_per_layer]})
    perm = [0, 2, 1, 3, 4, 6, 5, 7, 8]
    df_pareto = df_pareto.iloc[perm].reset_index(drop=True)

    unique_bit_widths = np.unique(bit_widths_pareto_per_layer)
    min_bit_width = min(unique_bit_widths)
    max_bit_width = max(unique_bit_widths)
    bin_width = (max_bit_width - min_bit_width) / 10
    bins = np.arange(min_bit_width, max_bit_width + bin_width, bin_width)

    num_layers = len(layer_names_per_index)
    fig, axs = plt.subplots(1, num_layers, figsize=(13, 2), sharey=True)

    for i, row in df_pareto.iterrows():
        layer = row['Layer']
        bit_widths = row['Bit Widths']
        axs[i].hist(bit_widths, bins=bins, color='#0065a7')
        axs[i].set_title(layer)
        axs[i].set_xticks(unique_bit_widths)
        axs[i].set_xticklabels(unique_bit_widths, ha="center")
        axs[i].grid(True)
        if i == 0:
            axs[i].set_ylabel('Frequency')
            axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.text(0.5, 0.02, "Bit-width per function", ha="center")
    plt.tight_layout()
    plt.savefig(f"{args.log_dir}/{log_file_name}_pareto_plot.pdf")
    plt.show()
