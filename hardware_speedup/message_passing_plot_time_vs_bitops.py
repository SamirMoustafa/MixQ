import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14

data_amd = pd.read_csv('message_passing_speedup_AMD_EPYC_9534.csv')
data_apple = pd.read_csv('message_passing_speedup_AppleM1-8-CoreGPU.csv')
data_intel = pd.read_csv('message_passing_speedup_IntelXeon-GoogleColabTPUv2.csv')

data_amd['Hardware'] = 'AMD EPYC 9534'
data_apple['Hardware'] = 'Apple M1 8-Core GPU'
data_intel['Hardware'] = 'Intel(R) Xeon(R) CPU'

combined_data = pd.concat([data_amd, data_apple, data_intel])

bit_size_mapping = {
    'torch.int8': 8,
    'torch.int16': 16,
    'torch.int32': 32,
    'torch.float32': 32
}
combined_data['Bit Size'] = combined_data['Data Type'].map(bit_size_mapping)
combined_data['Bit Operations'] = combined_data['Number of Operations'] * combined_data['Bit Size']

plt.figure(figsize=(6, 3))
colors = {'AMD EPYC 9534': '#885078', 'Apple M1 8-Core GPU': '#0065a7', 'Intel(R) Xeon(R) CPU': '#c76cc2'}
for hardware, group_data in combined_data.groupby('Hardware'):
    plt.scatter(group_data['Bit Operations'], group_data['Execution Time (s)'], label=hardware, color=colors[hardware])

    slope, intercept, r_value, p_value, std_err = linregress(group_data['Bit Operations'], group_data['Execution Time (s)'])
    x = np.linspace(group_data['Bit Operations'].min(), group_data['Bit Operations'].max(), 100)
    y = slope * x + intercept
    plt.plot(x, y, '--', color=colors[hardware])
    # Compute correlation coefficient
    print(f"Hardware: {hardware} - "
          f"Correlation Coefficient: {pearsonr(group_data['Bit Operations'], group_data['Execution Time (s)'])[0]:.3f} - "
          f"P-value: {pearsonr(group_data['Bit Operations'], group_data['Execution Time (s)'])[1]:.1e}")

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Bit Operations')
plt.ylabel('Inference Time (s)')
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig('message_passing_time_vs_bit_operations.pdf')
plt.show()