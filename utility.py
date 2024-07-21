import warnings
from io import BytesIO

import statistics
from os.path import isfile
from csv import DictWriter, reader

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from scipy.stats import skew, kurtosis, gaussian_kde

import plotly.graph_objects as go

from torch import meshgrid, arange


def format_title(title, subtitle=None, subtitle_font_size=14):
    title = f"<b>{title}</b>"
    if not subtitle:
        return title
    subtitle = f"<span style='font-size: {subtitle_font_size}px;'>{subtitle}</span>"
    return f"{title}<br>{subtitle}"


custom_template = {
    "layout": go.Layout(
        font={
            "family": "Nunito",
            "size": 16,
            "color": "#1f1f1f",
        },
        title={
            "font": {
                "family": "Lato",
                "size": 19,
                "color": "#1f1f1f",
            },
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        colorway=["#0065a7", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
        hovermode="closest",
    )
}


def plot_hist_statistics(x, title, show_pdf=False, show_cdf=False, remove_outliers=True, show=True, save_path=None):
    png_buffer = BytesIO()

    flatten_x = x.flatten().detach().cpu().numpy()
    if remove_outliers:
        q_low, q_high = np.percentile(flatten_x, [1, 99])
        flatten_x = flatten_x[(flatten_x > q_low) & (flatten_x < q_high)]

    fig = go.Figure()
    bins = min(100, max(1000, len(flatten_x) // 100))
    counts, bin_edges = np.histogram(flatten_x, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig.add_trace(go.Bar(x=bin_centers, y=counts, width=np.diff(bin_edges), name="Histogram", marker=dict(opacity=0.6)))

    if show_pdf:
        kde = gaussian_kde(flatten_x, bw_method="silverman")
        pdf = kde.pdf(bin_centers)
        fig.add_trace(go.Scatter(x=bin_centers, y=pdf, mode="lines", name="PDF", fill="tozeroy", marker=dict(opacity=0.6)))

    if show_cdf:
        fig.add_trace(go.Scatter(x=np.sort(flatten_x), y=np.linspace(0, 1, len(flatten_x)), mode="lines", name="CDF"))

    mean, median, std, skewness, kurt = np.mean(flatten_x), np.median(flatten_x), np.std(flatten_x), skew(flatten_x), kurtosis(flatten_x)
    stats_dict = {"Mean": mean,
                  "Median": median,
                  "Std": std,
                  "Skewness": skewness,
                  "Kurtosis": kurt,
                  "snr": abs(mean) / std,
                  "max": np.max(flatten_x),
                  "min": np.min(flatten_x),
                  }
    stats_text = "<br>".join([f"<b>{k}</b>: {v:.2f}" for k, v in stats_dict.items()])
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=0), showlegend=True, name=stats_text))
    png_buffer.seek(0)
    fig.update_layout(title=format_title(title),
                      xaxis_title="Values",
                      yaxis_title="Density",
                      template=custom_template,
                      margin={"l": 60, "r": 0, "t": 30, "b": 50},
                      )
    if save_path:
        fig.write_image(save_path)
    if show:
        png_buffer = BytesIO()
        fig.write_image(png_buffer, format="png")
        png_buffer.seek(0)
        image = Image.open(png_buffer)
        plt.tight_layout()
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def plot_2d_tensor_magnitude(x, title, xaxis_title="Node", yaxis_title="Features", show=True, save_path=None):
    if x.dim() != 2:
        raise ValueError("Tensor must be 2D")

    if x.size(0) > 100_000:
        warnings.warn("Too many elements to plot, subsampling the first 100k elements from the 0th dimension")
        x = x[:100_000]
    rows, cols = x.size()
    x_index, y_index = meshgrid(arange(rows), arange(cols), indexing="ij")
    z_values = x.abs().detach().cpu().numpy()

    custom_colorscale = [[0, "#0065a7",], [1, "#ff7f0e"]]
    fig = go.Figure(data=[go.Surface(z=z_values,
                                     x=x_index.numpy(),
                                     y=y_index.numpy(),
                                     colorscale=custom_colorscale,
                                     showscale=False,
                                     opacity=0.7,
                                     )
                          ])
    fig.update_layout(title=format_title(title, subtitle_font_size=18),
                      autosize=False,
                      scene=dict(xaxis=dict(title=xaxis_title, tickfont=dict(size=12)),
                                 yaxis=dict(title=yaxis_title, tickfont=dict(size=12)),
                                 zaxis=dict(title="Magnitude", tickfont=dict(size=12)),
                                 camera=dict(eye=dict(x=1.75, y=1.75, z=1.25)),
                                 aspectmode="cube",
                                 ),
                      template=custom_template,
                      margin={"l": 0, "r": 0, "t": 35, "b": 0},
                      height=800,
                      width=800,
                      )
    if save_path:
        fig.write_image(save_path)
    if show:
        png_buffer = BytesIO()
        fig.write_image(png_buffer, format="png")
        png_buffer.seek(0)
        image = Image.open(png_buffer)
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def flatten_list(nested_list):
    flattened_list = []

    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)

    return flattened_list


def flatten_tuple(t):
    for item in t:
        if isinstance(item, tuple):
            yield from flatten_tuple(item)
        else:
            yield item

def nested_median(*nested_lists):
    if not nested_lists:
        return []
    if isinstance(nested_lists[0], list):
        zipped = zip(*nested_lists)
        return [nested_median(*items) for items in zipped]
    else:
        values = [value for value in nested_lists if value is not None]
        if not values:
            return None
        return statistics.median_high(values)


def nested_max(*nested_lists):
    if not nested_lists:
        return []
    if isinstance(nested_lists[0], list):
        zipped = zip(*nested_lists)
        return [nested_max(*items) for items in zipped]
    else:
        values = [value for value in nested_lists if value is not None]
        if not values:
            return None
        return max(values)


def nested_min(*nested_lists):
    if not nested_lists:
        return []
    if isinstance(nested_lists[0], list):
        zipped = zip(*nested_lists)
        return [nested_min(*items) for items in zipped]
    else:
        values = [value for value in nested_lists if value is not None]
        if not values:
            return None
        return min(values)


def nested_std(*nested_lists):
    if not nested_lists:
        return []
    if isinstance(nested_lists[0], list):
        zipped = zip(*nested_lists)
        return [nested_std(*items) for items in zipped]
    else:
        filtered_values = [value for value in nested_lists if value is not None]
        if len(filtered_values) < 2:
            return 0
        return round(statistics.stdev(filtered_values), 3)


def format_fraction(number, max_precision=8):
    if '.' in f"{number:.{max_precision}f}":
        return f"{number:.{max_precision}f}".rstrip('0').rstrip('.')
    return f"{number:.{max_precision}f}"


def write_to_csv(file_path, data):
    file_exists = isfile(file_path)
    headers = data.keys()
    with open(file_path, mode='a', newline='') as file:
        writer = DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        else:
            with open(file_path, mode='r') as read_file:
                r = reader(read_file)
                existing_headers = next(r)
                if set(existing_headers) != set(headers):
                    raise ValueError("CSV file headers do not match the data dict keys")
        writer.writerow(data)