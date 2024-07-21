import os
import subprocess
import time

import numpy as np

# Environment setup
conda_base = os.getenv('CONDA_PREFIX', '/home/opt/conda')  # adjust as needed
activate_script = os.path.join(conda_base, 'etc', 'profile.d', 'conda.sh')

# Conda environment to activate
env_name = 'py311'


# Define the command to activate conda environment
def get_source_command():
    return f". {activate_script} && conda activate {env_name} && "


# List of lambda values
lambda_values = [-1e-8, ] + np.linspace(0, 1, 9).tolist()
n = 10

# Running scripts in batches
for i in range(0, len(lambda_values), n):
    processes = []
    for j in range(n):
        index = i + j
        if index < len(lambda_values):
            lambda_value = lambda_values[index]
            print(f"Running script with bit_width_lambda = {lambda_value}")
            command = f"{get_source_command()}python gin_tudataset_mixed_q.py --bit_width_lambda={lambda_value}"
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                executable='/bin/bash'
            )
            processes.append(process)
        time.sleep(1)

    # Wait and print output/errors
    for process in processes:
        stdout, stderr = process.communicate()
        if process.returncode != 0:  # If the process failed
            print("Process failed with return code:", process.returncode)
            print("Output:", stdout.decode())
            print("Error:", stderr.decode())
