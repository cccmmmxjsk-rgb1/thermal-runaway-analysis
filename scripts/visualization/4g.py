import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "gas_correlation_data.xlsx")

df_raw = pd.read_excel(file_path)

target_gases = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']
df = df_raw[target_gases].copy()

def format_gas_name(name):
    formatted_name = re.sub(r'(\d+)', r'_{\1}', name)
    return f"${formatted_name}$"

# Format gas names for display
df.rename(columns=lambda x: format_gas_name(x), inplace=True)

# Compute the correlation matrix
df_inverted = 1 - df
corr = df_inverted.corr()
corr_reversed = corr.iloc[::-1]

mask = np.triu(np.ones_like(corr_reversed, dtype=bool), k=0)
mask = np.flip(mask, axis=1)

sns.set_theme(style="white")
plt.figure(figsize=(10, 9))

heatmap = sns.heatmap(
    corr_reversed,
    mask=mask,
    cmap='RdBu_r',
    vmax=1,
    vmin=-1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar=False,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 16}
)

ax = plt.gca()
ax.yaxis.tick_right()
ax.xaxis.tick_bottom()

ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

ax.tick_params(
    axis='x',
    which='both',
    bottom=True,
    top=False,
    labelbottom=True,
    direction='out',
    color='black',
    width=2,
    length=6
)
ax.tick_params(
    axis='y',
    which='both',
    left=False,
    right=True,
    labelright=True,
    direction='out',
    color='black',
    width=2,
    length=6
)

plt.xticks(rotation=0, fontsize=26, color='black')
plt.yticks(rotation=0, fontsize=26, color='black')

plt.tight_layout()
plt.show()
