import inspect
import os
from collections import OrderedDict

from torch import is_tensor, zeros, int8, uint8, int16, int32, int64, float16, bfloat16, float32, float64
from torch.nn import Module

msg_special_args = {"edge_index", "edge_index_i", "edge_index_j", "size", "size_i", "size_j"}
aggr_special_args = {"index", "dim_size"}


def __process_size__(size):
    if isinstance(size, int):
        return [size, size]
    if is_tensor(size):
        return size.tolist()
    return list(size) if size else [None, None]


def __distribute__(params, kwargs):
    return {key: kwargs.get(key, param.default) for key, param in params.items()}


class MessagePassing(Module):
    def __init__(self, dtype, flow: str = "source_to_target", node_dim: int = 0):
        super(MessagePassing, self).__init__()
        assert dtype in [int8, uint8, int16, int32, int64, float16, bfloat16, float32, float64], f"{dtype} is not a valid data type."
        assert flow in ["source_to_target", "target_to_source"], f"{flow} is not a valid flow direction."
        assert node_dim >= 0 and isinstance(node_dim, int), "node_dim must be non-negative integer."

        self.dtype = dtype
        self.flow = flow
        self.node_dim = node_dim

        self.__msg_params__ = OrderedDict(inspect.signature(self.message).parameters)
        self.__aggr_params__ = OrderedDict(inspect.signature(self.aggregate).parameters)
        self.__aggr_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        self.__args__ = msg_args.union(aggr_args)

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            idx = ij.get(arg[-2:])
            data = kwargs.get(arg[:-2] if idx is not None else arg)

            if idx is not None and is_tensor(data):
                size[idx] = data.shape[self.node_dim]
                out[arg] = data.index_select(self.node_dim, edge_index[idx])
            else:
                out[arg] = data

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        out.update({"edge_index": edge_index,
                    "edge_index_i": edge_index[i],
                    "edge_index_j": edge_index[j],
                    "size": size,
                    "size_i": size[i],
                    "size_j": size[j],
                    "index": edge_index[i],
                    "dim_size": size[i]})
        return out

    def forward(self, edge_index, size=None, **kwargs):
        size = __process_size__(size)
        kwargs = self.__collect__(edge_index, size, kwargs)
        out = self.message(**__distribute__(self.__msg_params__, kwargs))
        out = self.aggregate(out, **__distribute__(self.__aggr_params__, kwargs))
        return out if not isinstance(out, tuple) else out[0]

    def aggregate(self, inputs, index, dim_size):
        num_features = inputs.shape[1]
        index = index.view(-1, 1).expand(-1, num_features) if inputs.dim() > 1 else index
        return zeros((dim_size, num_features),
                     device=inputs.device,
                     dtype=self.dtype).scatter_add_(self.node_dim, index, inputs)

    def __repr__(self):
        return "{}(dtype={})".format(self.__class__.__name__, self.dtype)

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


if __name__ == "__main__":
    def pin_to_core(core_id=0):
        try:
            from os import sched_setaffinity
            sched_setaffinity(0, {core_id})
        except AttributeError:
            print("sched_setaffinity is not supported on this platform.")

    # Comment out the following line if you are not using a linux machine
    pin_to_core()

    from time import perf_counter, sleep

    from numpy import mean, std, arange
    from pandas import DataFrame, concat
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams["font.size"] = 28

    from prettytable import PrettyTable
    from torch import long, iinfo
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    from torch_geometric.data import Batch
    from torch_geometric.datasets import Planetoid

    from transormed_tudataset import TransformedTUDataset

    from torch_operation_counter import OperationsCounterMode


    def quantize(x, dtype, device="cpu"):
        if dtype in [float16, bfloat16, float32, float64]:
            return x.to(device=device, dtype=dtype)

        q_min, q_max = iinfo(dtype).min, iinfo(dtype).max
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min).item() / (q_max - q_min)
        x_q = x / scale
        x_q = x_q.clamp(q_min, q_max).round().to(device=device, dtype=dtype)
        return x_q


    dtypes = [int8, int16, int32, float32]

    datasets_names = ["Cora", "Citeseer", "Pubmed", "ogbn-arxiv", "IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "DD", "REDDIT-BINARY", "REDDIT-MULTI-5K"]
    df = DataFrame(columns=["Dataset", "Data Type", "Execution Time (s)", "Execution Time (s) std", "Number of Operations", "Speedup", "Speedup std", "Runtime Reduction", "Runtime Reduction std"])

    for dataset_name_i in datasets_names:
        if dataset_name_i in ["Cora", "Citeseer", "Pubmed"]:
            data = Planetoid(root='./data/', name=dataset_name_i)[0]
        elif dataset_name_i in ["ogbn-arxiv"]:
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='./data/')
            data = dataset[0]
        elif dataset_name_i in ["IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "DD", "REDDIT-BINARY", "REDDIT-MULTI-5K"]:
            dataset = TransformedTUDataset(root='./data/', name=dataset_name_i, use_node_attr=True, use_edge_attr=True)
            data = Batch.from_data_list([dataset[i] for i in range(len(dataset))])
        else:
            raise ValueError("Invalid dataset name")

        n = data.num_nodes
        e = data.num_edges
        num_feature = data.num_features

        runtime_per_dtype = {dtype: dict() for dtype in dtypes}
        for dtype in dtypes:
            x = quantize(data.x, dtype)
            edge_index = data.edge_index.to(dtype=long)
            edge_weight = quantize(data.edge_attr, dtype) if data.edge_attr is not None else None

            timing_list = []
            sleep(1)
            cov = MessagePassing(dtype=dtype)

            with OperationsCounterMode(cov) as ops_counter:
                cov(edge_index, size=(n, n), x=x, edge_weight=edge_weight)

            for _ in range(3):
                cov(edge_index, size=(n, n), x=x, edge_weight=edge_weight)
            for _ in range(10):
                start_time = perf_counter()
                cov(edge_index, size=(n, n), x=x, edge_weight=edge_weight)
                end_time = perf_counter()
                timing_list.append(end_time - start_time)
            runtime_per_dtype[dtype] = {"mean": mean(timing_list), "std": std(timing_list)}

        prettytable = PrettyTable()
        prettytable.title = f"Dataset: {dataset_name_i} | OPs: {ops_counter.total_operations} (n={n}, e={e}, f={num_feature})"
        prettytable.field_names = ["Data Type", "Execution Time (s)", "Speedup", "Runtime Reduction"]
        fp32_time = runtime_per_dtype[float32]["mean"]
        for dtype, execution_time in runtime_per_dtype.items():
            speedup = fp32_time / execution_time["mean"]
            speedup_std = speedup * (execution_time["std"] / execution_time["mean"])
            runtime_reduction = (fp32_time - execution_time["mean"]) / fp32_time * 100
            runtime_reduction_std = runtime_reduction * (execution_time["std"] / execution_time["mean"])
            prettytable.add_row([dtype,
                                 f"{execution_time['mean']:.6f} ± {execution_time['std']:.6f}",
                                 f"{speedup:.2f}x",
                                 f"{runtime_reduction:.2f}%"])
            df = concat([df, DataFrame([[dataset_name_i,
                                         dtype,
                                         execution_time["mean"],
                                         execution_time["std"],
                                         ops_counter.total_operations,
                                         speedup,
                                         speedup_std,
                                         runtime_reduction,
                                         runtime_reduction_std,
                                         ]],
                                       columns=["Dataset",
                                                "Data Type",
                                                "Execution Time (s)",
                                                "Execution Time (s) std",
                                                "Number of Operations",
                                                "Speedup",
                                                "Speedup std",
                                                "Runtime Reduction",
                                                "Runtime Reduction std"
                                                ])])
        print(prettytable)

    print(df.to_string())
    df.to_csv("message_passing_speedup.csv", index=False)

    datasets = df['Dataset'].unique()
    data_types = df['Data Type'].unique()

    def plot_speedup(title):
        color_palette = ["#0065a7", "#885078", "#c76cc2", "#f0a6ca"]
        plt.figure(figsize=(13, 5))
        width = 0.2

        for i, dtype in enumerate(data_types):
            if dtype == float32:
                continue
            subset = df[(df['Data Type'] == dtype)]
            x_positions = arange(len(datasets)) + i * width
            plt.bar(x_positions, subset['Speedup'], width=width, label=dtype, yerr=subset['Speedup std'], capsize=4, ecolor="gray", color=color_palette[i])

        plt.xlabel("Dataset Name")
        plt.ylabel("Speedup")
        # Replace Dataset names with shorter names
        datasets_new_names = {"ogbn-arxiv": "ArXiv", "IMDB-BINARY": "IMDB-B", "IMDB-MULTI": "IMDB-M", "REDDIT-BINARY": "REDDIT-B", "REDDIT-MULTI-5K": "REDDIT-M", "DD": "D&D"}
        datasets_names = [datasets_new_names.get(d, d) for d in datasets]
        plt.xticks(arange(len(datasets_names)) + width * (len(data_types) / 4), datasets_names, fontsize=16)
        plt.yticks(arange(1, max(df['Speedup']) + 1, 1), [f"{i:.0f}x" for i in arange(1, max(df['Speedup']) + 1, 1)])
        plt.legend(title="Data Type", title_fontsize="22", fontsize="22", loc="upper right")
        plt.title(title)
        plt.tight_layout()
        plt.grid(axis="y")
        plt.savefig(f"message_passing_speedup.pdf")
        plt.show()


    # Plot for CPU
    cpu_name = os.popen("lscpu | grep 'Model name'").read().split(":")[1].strip()
    plot_speedup(cpu_name)
