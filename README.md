<div align="center">
<img src="https://i.imgur.com/arMZl3N_d.webp?maxwidth=760&fidelity=grand" width="200">
<h1> Mixed Precision Quantization in Graph Neural Networks (MixQ-GNN)

![python-3.9](https://img.shields.io/badge/python-3.11.5-blue)
![license](https://img.shields.io/badge/license-MIT-green)
_________________________
</div>

This is the official repository for the paper "Efficient Mixed Precision Quantization in Graph Neural Networks".

## Getting Started
1. Clone or download the repository.
    ```
    git clone https://github.com/SamirMoustafa/MixQ.git
    cd MixQ
    ```
2. To get started with the project, there are two ways:
   * Use the provided Docker image.
   ```bash
   docker build -t mixq .
   docker run --gpus all --rm -ti --ipc=host --name mixq_instance mixq /bin/bash
   ```
   * Or, install the required dependencies manually through anaconda.
   ```bash
   conda create -n mixq python=3.11.5
   conda activate mixq
   # Install PyTorch depending on the current machine setup.
   pip install numpy==1.26.4
   command -v nvidia-smi > /dev/null && conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia || conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch
   # Install PyG dependencies based on the current PyTorch setup
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f $(python -c "import torch; print('https://data.pyg.org/whl/torch-2.2.1+cu121.html' if torch.cuda.is_available() else 'https://data.pyg.org/whl/torch-2.2.1+cpu.html')")
   export PYTHONPATH="${PYTHONPATH}:./"
   pip install -r requirements.txt
   ```
3. (Optional) Verify the installation by running the following command:
   ```bash
   python -m unittest discover ./test
   ```
4. (Optional) Verify `Quantized Message Passing Schema` theorem only for GCN and GIN examples by running the following commands:
   ```bash
   cd test/
   export PYTHONPATH="${PYTHONPATH}:../"
   python -m unittest ./test_graph_conv_module.py
   python -m unittest ./test_graph_iso_module.py 
   ```
## Reproduce Results
   * Tasks per Node
     * Figure 1 can be reproduced by running the following command:
     ```bash
        python tasks_per_node/planetoid/plot_ops_vs_acc.py
        ```
     * Figure 2 and Figure 3 can be reproduced by running the following command:
     ```bash
     python tasks_per_node/planetoid/gcn_planetoid_explore_all.py --dataset_name Cora
     ```
     * Figure 6 can be reproduced by running the following command:
     ```bash
     python tasks_per_node/planetoid/gcn_planetoid_run_experiments.py
     ```
     * Table 1 can be reproduced by running the following commands:
     ```bash
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name Cora --bit_width_lambda -0.000000001
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name Cora --bit_width_lambda 0.1
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name Cora --bit_width_lambda 1.0
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name CiteSeer --bit_width_lambda -0.000000001
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name CiteSeer --bit_width_lambda 0.1
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name CiteSeer --bit_width_lambda 1.0
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name PubMed --bit_width_lambda -0.000000001
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name PubMed --bit_width_lambda 0.1
     python tasks_per_node/planetoid/gcn_planetoid.py --dataset_name PubMed --bit_width_lambda 1.0
     python tasks_per_node/ogbn/gcn_with_ogb.py --dataset_name ogbn-arxiv --bit_width_lambda -0.000000001
     python tasks_per_node/ogbn/gcn_with_ogb.py --dataset_name ogbn-arxiv --bit_width_lambda 0.1
     python tasks_per_node/ogbn/gcn_with_ogb.py --dataset_name ogbn-arxiv --bit_width_lambda 1.0
     ```
     * Table 2 can be reproduced by running the following commands:
     ```bash
     python tasks_per_node/planetoid/gcn_planetoid_with_degree_quant.py --dataset_name Cora --bit_width_lambda -0.000000001
     python tasks_per_node/planetoid/gcn_planetoid_with_degree_quant.py --dataset_name Cora --bit_width_lambda 0.1
     python tasks_per_node/planetoid/gcn_planetoid_with_degree_quant.py --dataset_name Cora --bit_width_lambda 1.0
     ```
     * Table 4 can be reproduced by running the following commands:
     ```bash
     python tasks_per_node/planetoid/gcn_planetoid_random_bit_width.py
     ```
* Tasks per Graph
  * FP32 results for TUDataset in Table 3 can be reproduced by running the following commands:
  ```bash
  python examples/gin_tudataset_fp32.py --dataset_name IMDB-BINARY
  python examples/gin_tudataset_fp32.py --dataset_name PROTEINS
  python examples/gin_tudataset_fp32.py --dataset_name DD
  python examples/gin_tudataset_fp32.py --dataset_name REDDIT-BINARY
  ```
  * TUDataset results in Table 3 can be reproduced by running the following commands:
  ```bash
  # IMDB-BINARY
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda -0.00000001
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 0.0
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 0.125
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 0.25
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 0.375
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 0.5
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 0.625
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 0.75
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 0.875
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name IMDB-BINARY --bit_width_lambda 1.0
  ```
  ```bash
  # PROTEINS
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda -0.00000001
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 0.0
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 0.125
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 0.25
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 0.375
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 0.5
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 0.625
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 0.75
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 0.875
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name PROTEINS --bit_width_lambda 1.0
  ```
  ```bash
  # DD
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda -0.00000001
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 0.0
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 0.125
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 0.25
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 0.375
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 0.5
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 0.625
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 0.75
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 0.875
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name DD --bit_width_lambda 1.0
  ```
  ```bash
  # REDDIT-BINARY
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda -0.00000001
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 0.0
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 0.125
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 0.25
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 0.375
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 0.5
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 0.625
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 0.75
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 0.875
  python tasks_per_graph/tudataset/gin_tudataset_mixed_q.py --dataset_name REDDIT-BINARY --bit_width_lambda 1.0
  ```
  * Table 8 can be reproduced by running the following commands:
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
├── tasks_per_node/
│   ├── planetoid/
│   │   ├── explore_all_logs/
│   │   │   └── Cora/ (📈📉 Figure 2 and Figure 3)
│   │   ├── ablation_study/
│   │   │   ├── CiteSeer/ (📈📉 Figure 16)
│   │   │   ├── Cora/ (📈📉 Figure 6)
│   │   │   └── PubMed/ (📈📉 Figure 17)
│   │   ├── experimental_plus_DQ_logs/
│   │   │   ├── CiteSeer/ (📄📝 Table 6)
│   │   │   ├── Cora/ (📄📝 Table 2)
│   │   │   └── PubMed/ (📄📝 Table 6)
│   │   └── experimental_logs/
│   │       ├── CiteSeer/ (📄📝 Table 1 and Table 4)
│   │       ├── Cora/ (📄📝 Table 1 and Table 4)
│   │       └── PubMed/ (📄📝 Table 1 and Table 4)
│   └── ogbn/
│       └── experimental_logs/
│           └── ogbn-arxiv/ (📄📝 Table 1)
├── tasks_per_graph/
│   ├── synthetic/
│   │   └── logs/ (📄📝 Table 8)
│   └── tudataset/
│       ├── a2q_logs/ (📄📝 A^2Q results for TUDataset in Table 3 and Table 7)
│       ├── dq_logs/ (📄📝 DQ results for TUDataset in Table 3 and Table 7)
│       └── logs/ (📄📝 MixQ results for TUDataset in Table 3 and Table 7)
├── examples/
│   └── logs/ (📄📝 FP32 results for TUDataset in Table 3)
└── hardware_speedup/
    ├── bitBLAS_layout_nt_NVIDIA_A100_80GB_PCIe.csv (📈📉 Figure 14)
    ├── message_passing_speedup_AMD_EPYC_9534.csv (📈📉 Figure 13(b) and Figure 15)
    ├── message_passing_speedup_AppleM1-8-CoreGPU.csv (📈📉 Figure 13(c) and Figure 15)
    └── message_passing_speedup_IntelXeon-GoogleColabTPUv2.csv (📈📉 Figure 13(a) and Figure 15)
```

## Citation
```
@inproceedings{Moustafa2025MixQGNN,
  author={Samir, Moustafa and Nils, Kriege and  Wilfried, Gansterer},
  booktitle={2025 IEEE 41st International Conference on Data Engineering (ICDE)}, 
  title={Efficient Mixed Precision Quantization in Graph Neural Networks}, 
  year={2025},
  keywords={Graph neural networks, Quantization},
  }
```
