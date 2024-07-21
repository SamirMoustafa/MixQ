<div align="center">
<img src="https://i.imgur.com/BOj6R2A_d.webp?maxwidth=760&fidelity=grand" width="200">
<h1> Mixed Precision Quantization in Graph Neural Networks (MixQ-GNN)

![python-3.9](https://img.shields.io/badge/python-3.11.5-blue)
![license](https://img.shields.io/badge/license-MIT-green)
_________________________
</div>

This is the official repository for the paper "Efficient Mixed Precision Quantization in Graph Neural Networks".

## Getting Started
1. Clone or download the repository (due to the anonymous process, repository can be downloaded as zip file only).
2. To get started with the project, there are two ways:
   * Use the provided Docker image.
   ```bash
   docker build -t mixq .
   docker run --gpus all --rm -ti --ipc=host --name mixq_instance mixq /bin/bash
   ```
   * Install the required dependencies manually through anaconda.
   ```bash
   conda create -n mixq python=3.11.5
   conda activate mixq
   # Install PyTorch version 2.2.1 that is compatible with the target machine
   # Install torch_scatter version 2.1.2 that is compatible with PyTorch
   pip install -r requirements.txt
   ```
3. Verify the installation by running the following command:
   ```bash
   python -m unittest discover ./test
   ```
4. To reproduce the results, follow the instructions in the respective sections.
   * Tasks per Node
     * Figure 2 and Figure 3 can be reproduced by running the following commands:
     ```bash
     python tasks_per_node/gcn_planetoid_explore_all.py --dataset_name Cora
     ```
     * Figure 6 can be reproduced by running the following commands:
     ```bash
     python tasks_per_node/gcn_planetoid_run_experiments.py
     ```
     * Table 1 and Table 4 can be reproduced by running the following commands:
     ```bash
     python tasks_per_node/gcn_planetoid.py --dataset_name Cora --bit_width_lambda -0.000000001
     python tasks_per_node/gcn_planetoid.py --dataset_name Cora --bit_width_lambda 0.1
     python tasks_per_node/gcn_planetoid.py --dataset_name Cora --bit_width_lambda 1.0
     python tasks_per_node/gcn_planetoid.py --dataset_name CiteSeer --bit_width_lambda -0.000000001
     python tasks_per_node/gcn_planetoid.py --dataset_name CiteSeer --bit_width_lambda 0.1
     python tasks_per_node/gcn_planetoid.py --dataset_name CiteSeer --bit_width_lambda 1.0
     python tasks_per_node/gcn_planetoid.py --dataset_name PubMed --bit_width_lambda -0.000000001
     python tasks_per_node/gcn_planetoid.py --dataset_name PubMed --bit_width_lambda 0.1
     python tasks_per_node/gcn_planetoid.py --dataset_name PubMed --bit_width_lambda 1.0
     ```
     * Table 2 can be reproduced by running the following commands:
     ```bash
     python tasks_per_node/gcn_with_ogb.py
     python tasks_per_node/gcn_with_ogb_plus_degree_quant.py
     ```
   * Tasks per Graph
     * FP32 results for TUDataset in Table 3 can be reproduced by running the following commands:
     ```bash
     python examples/gin_tudataset_fp32.py
     ```
     * TUDataset results in Table 3 can be reproduced by running the following commands:
     ```bash
     python tasks_per_graph/tudataset/gin_tudataset_mixed_q_run_experiments.py
     ```
     * Table 6 can be reproduced by running the following commands:
     ```bash
     python tasks_per_graph/synthetic/synthetic_pyg_run.py
     python tasks_per_graph/synthetic/synthetic_q_run.py --bit_width 1
     python tasks_per_graph/synthetic/synthetic_q_run.py --bit_width 2
     python tasks_per_graph/synthetic/synthetic_q_run.py --bit_width 4
     python tasks_per_graph/synthetic/synthetic_mixed_q_run.py --bit_width_lambda -0.0001
     python tasks_per_graph/synthetic/synthetic_mixed_q_run.py --bit_width_lambda 0.0
     python tasks_per_graph/synthetic/synthetic_mixed_q_run.py --bit_width_lambda 0.0001
     ```
     

## Logs Directories of the Experiments
```
./
├── examples/
│   └── logs/ (📄📝 FP32 results for TUDataset in Table 3)
├── hardware_speedup/
│   ├── bitBLAS_layout_nt_NVIDIA_A100_80GB_PCIe.csv (📈📉 Figure 12)
│   ├── message_passing_speedup_AMD_EPYC_9534.csv (📈📉 Figure 11(b) and Figure 13)
│   ├── message_passing_speedup_AppleM1-8-CoreGPU.csv (📈📉 Figure 11(c) and Figure 13)
│   └── message_passing_speedup_IntelXeon-GoogleColabTPUv2.csv (📈📉 Figure 11(a) and Figure 13)
├── tasks_per_graph/
│   ├── synthetic/
│   │   └── logs/ (📄📝 Table 6)
│   └── tudataset/
│       ├── a2q_logs/ (📄📝 A^2Q results for TUDataset in Table 3)
│       ├── dq_logs/ (📄📝 DQ results for TUDataset in Table 3)
│       └── logs/ (📄📝 MixQ results for TUDataset in Table 3)
└── tasks_per_node/
    ├── explore_all_logs/
    │   └── Cora/ (📈📉 Figure 2 and Figure 3)
    ├── ablation_study/
    │   ├── CiteSeer/ (📈📉 Figure 14)
    │   ├── Cora/ (📈📉 Figure 6)
    │   └── PubMed/ (📈📉 Figure 15)
    └── experimental_logs/
        ├── CiteSeer/ (📄📝 Table 1 and Table 4)
        ├── Cora/ (📄📝 Table 1 and Table 4)
        ├── PubMed/ (📄📝 Table 1 and Table 4)
        └── ogbn-arxiv/ (📄📝 Table 2)

```

## Directory Structure
```
./
├── data/
│   └── CSL (Dataset raw data)
├── examples/
│   ├── cnn_mnist_fixed_q_freezable.py
│   ├── gcn_planetoid_q_freezable.py
│   ├── gin_mnist_q_freezable.py
│   ├── gin_tudataset_degree_quant.py
│   ├── gin_tudataset_fp32.py
│   ├── gin_tudataset_q.py
│   ├── linear_mnist_mixed_q.py
│   ├── linear_mnist_q_freezable.py
│   └── mix_q_freezable_demo.py
├── hardware_speedup/
│   ├── bitBLAS_example.py
│   ├── bitBLAS_plot_csv.py
│   ├── message_passing_plot_time_vs_bitops.py
│   └── message_passing_with_diff_precision.py
├── quantization/
│   ├── __init__.py
│   ├── base_parameter.py
│   ├── functional.py
│   ├── message_passing_base.py
│   ├── fixed_modules/
│   │   ├── __init__.py
│   │   ├── base_module.py
│   │   └── non_parametric/
│   │       ├── __init__.py
│   │       ├── activations.py
│   │       ├── arbitrary_function.py
│   │       ├── input_quantizer.py
│   │       ├── max_pooling.py
│   │       └── message_passing.py
│   ├── parametric/
│   │   ├── __init__.py
│   │   ├── convolution.py
│   │   ├── graph_convolution.py
│   │   ├── graph_isomorphism.py
│   │   └── linear.py
│   ├── mixed_modules/
│   │   ├── __init__.py
│   │   ├── base_module.py
│   │   └── non_parametric/
│   │       ├── __init__.py
│   │       ├── activations.py
│   │       ├── arbitrary_function.py
│   │       ├── input_quantizer.py
│   │       └── message_passing.py
│   ├── parametric/
│   │   ├── __init__.py
│   │   ├── graph_convolution.py
│   │   ├── graph_isomorphism.py
│   │   └── linear.py
│   └── utility.py
├── tasks_per_graph/
│   ├── synthetic/
│   │   ├── CSL.py
│   │   ├── loader.py
│   │   ├── synthetic_mixed_q_run.py
│   │   ├── synthetic_pyg_run.py
│   │   ├── synthetic_q_run.py
│   │   └── synthetic_utility.py
│   └── tudataset/
│       ├── gin_tudataset_mixed_q.py
│       └── gin_tudataset_mixed_q_run_experiments.py
├── tasks_per_node/
│   ├── gcn_planetoid.py
│   ├── gcn_planetoid_explore_all.py
│   ├── gcn_planetoid_random_bit_width.py
│   ├── gcn_planetoid_run_experiments.py
│   ├── gcn_with_ogb.py
│   ├── gcn_with_ogb_plus_degree_quant.py
│   └── plot_ablation_study.py
├── test/
│   ├── test_conv2d_module.py
│   ├── test_graph_conv_module.py
│   ├── test_graph_iso_module.py
│   ├── test_linear_module.py
│   ├── test_message_passing.py
│   └── utils.py
├── training/
│   ├── __init__.py
│   ├── logger.py
│   ├── probability_degree_transforms.py
│   ├── tensorboard_logger.py
│   └── train_evaluate.py
├── README.md
├── Dockerfile
├── requirements.txt
├── transormed_tudataset.py
└── utility.py
```
