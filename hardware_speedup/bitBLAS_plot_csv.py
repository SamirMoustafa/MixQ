import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


if __name__ == '__main__':
    df = pd.read_csv('./bitBLAS_layout_nt_NVIDIA_A100_80GB_PCIe.csv')
    dataset_names_to_add_y_label = ['CiteSeer', 'IMDB-B', 'REDDIT-M']

    colums_to_be_sorted = ['A_dtype', 'W_dtype', 'Accum_dtype', 'Out_dtype']
    custom_order = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'int4', 'nf4', 'int2']
    for column in colums_to_be_sorted:
        df[column] = pd.Categorical(df[column], categories=custom_order, ordered=True)

    df_sorted = df.sort_values(by=list(colums_to_be_sorted))

    dtype_map = {'float64': 'FP64',
                 'float32': 'FP32',
                 'float16': 'FP16',
                 'int64': 'INT64',
                 'int32': 'INT32',
                 'int16': 'INT16',
                 'int8': 'INT8',
                 'int4': 'INT4',
                 'nf4': 'NF4',
                 'int2': 'INT2',
                 }
    df['A_dtype'] = df['A_dtype'].map(dtype_map)
    df['W_dtype'] = df['W_dtype'].map(dtype_map)
    df['Accum_dtype'] = df['Accum_dtype'].map(dtype_map)
    df['Out_dtype'] = df['Out_dtype'].map(dtype_map)

    colors = ["#0065a7", "#885078", "#c76cc2", "#f0a6ca"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

    for dataset_i in df["Dataset"].unique():
        # Creating the pivot table
        pivot_table = df[df["Dataset"] == dataset_i].pivot_table(
            values='Latency (ms)',
            index=['A_dtype', 'W_dtype'],
            columns=['Accum_dtype', 'Out_dtype'],
            aggfunc='median'
        )

        num_cells = pivot_table.shape[0] * pivot_table.shape[1]
        font_size = 500 // num_cells
        # Plotting the heatmap
        plt.figure(figsize=(5, 5))
        annot_kws = {'fontsize': 5,
                     'rotation': 45,
                     'verticalalignment': 'center',
                     'font': 'STIXGeneral',
                     }
        ax = sns.heatmap(pivot_table, annot=True, cmap=cmap, fmt=".4f", annot_kws=annot_kws, linewidths=1)

        ax.set_xlabel('$\\text{Accumulated}(X\Theta)_{\\text{dtype}} - X\Theta_{\\text{dtype}}$', fontsize=24)
        ax.set_ylabel('$X_{\\text{dtype}} - \Theta_{\\text{dtype}}$', fontsize=24)
        if dataset_i in dataset_names_to_add_y_label:
            ax.figure.axes[-1].set_ylabel('Latency (ms)', size=24)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=24)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.get_offset_text().set(size=18)

        # Optional: Customizing tick labels if needed
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(f'./dense_{dataset_i}_speedup_heatmap.pdf')
        plt.show()