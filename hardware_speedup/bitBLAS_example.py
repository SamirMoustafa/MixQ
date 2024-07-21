import os

import pandas as pd
from tqdm import tqdm

from bitblas.utils.target_detector import auto_detect_nvidia_target
from bitblas import Matmul, MatmulConfig


# Initialize possible values for data types
A_dtypes = ["float16", "float32", "float64", "int32", "int8"]
W_dtypes = ["float16", "float32", "float64", "int32", "int8", "int4", "int2", "int1", "nf4", "fp4_e2m1"]
accum_dtypes = ["float16", "int32"]
out_dtypes = ["float16", "float32", "int32", "int8"]

# Set default values
target = auto_detect_nvidia_target()
layout = "nt"  # "nt" or "tn"
datasets_test_shapes = {
    "Cora": (1433, 128, 128),
    "CiteSeer": (3703, 128, 128),
    "PubMed": (500, 128, 128),
    "ArXiv": (128, 256, 256),
    "IMDB-B": (136, 128, 128),
    "PROTEINS": (4, 128, 128),
    "DD": (89, 128, 128),
    "REDDIT-B": (1, 128, 128),
    "REDDIT-M": (1, 128, 128),
}

# DataFrame to store results
results_df = pd.DataFrame(columns=["A_dtype", "W_dtype", "Out_dtype", "Accum_dtype", "Matrix Shape", "Latency (ms)"])

# Iterate over all combinations of data types
pbar = tqdm([(A_dtype, W_dtype, accum_dtype, out_dtype) for A_dtype in A_dtypes
            for W_dtype in W_dtypes for accum_dtype in accum_dtypes for out_dtype in out_dtypes])
for activation_dtype, x_dtype, accum_dtype, y_dtype in pbar:
    for dataset, shapes in datasets_test_shapes.items():
        M, N, K = shapes
        config = MatmulConfig(M, N, K, activation_dtype, x_dtype, y_dtype, accum_dtype, layout)
        try:
            matmul = Matmul(config, target=target, enable_tuning=True)
            kernel_latency = matmul.profile_latency()

            if matmul.input_transform is not None:
                kernel_latency += matmul.ladder_permutate_a.profile_latency()

            new_row = pd.DataFrame([{
                "A_dtype": activation_dtype,
                "W_dtype": x_dtype,
                "Out_dtype": y_dtype,
                "Accum_dtype": accum_dtype,
                "Dataset": dataset,
                "Matrix Shape": f"{M}x{N}x{K}",
                "Latency (ms)": f"{kernel_latency}"
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

        except Exception as e:
            print(f"Configuration: A_dtype={activation_dtype}, W_dtype={x_dtype}, Out_dtype={y_dtype}, Accum_dtype={accum_dtype}")

# Print DataFrame as a string
print(results_df.to_string(index=False))
nvidia_arch_name = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip().replace(" ", "_")
results_df.to_csv(f"./bitBLAS_layout_{layout}_{nvidia_arch_name}.csv", index=False)
