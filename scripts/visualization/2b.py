import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

# Global figure style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'

# Pricing database:
# [input price per 1M tokens, output price per 1M tokens, display name]
pricing_db = {
    'gpt5':             [1.25, 10.00, 'GPT-5'],
    'deepseekr1':       [0.55, 2.19,  'DeepSeek-R1'],
    'deepseek-v3':      [0.14, 0.28,  'DeepSeek-V3'],
    'claude4':          [3.00, 15.00, 'Claude-4'],
    'qwen-plus':        [0.40, 1.20,  'Qwen-Plus'],
    'GPT-4o':           [2.50, 10.00, 'GPT-4o'],
    'gemini2.5flash':   [0.05, 0.20,  'Gemini-2.5-Flash'],
    'qwen-trub':        [0.02, 0.06,  'Qwen-Turbo'],
}

# Model order used in the comparison
model_order = [
    'gpt5', 'deepseekr1', 'deepseek-v3', 'claude4',
    'qwen-plus', 'GPT-4o', 'gemini2.5flash', 'qwen-trub'
]

# Token usage per paper
TOKENS_IN, TOKENS_OUT = 15000, 3000

# Compute the estimated cost per paper for each model
data = []
for key in model_order:
    p_in, p_out, display_name = pricing_db[key]
    cost = (TOKENS_IN * p_in + TOKENS_OUT * p_out) / 1_000_000
    data.append({'Model': display_name, 'Price': cost})

df_price = pd.DataFrame(data)

# Sort models by price in ascending order
df_price = df_price.sort_values(by='Price', ascending=True).reset_index(drop=True)

# Compress x-axis spacing to create a compact layout
compact_ratio = 0.42
x_positions = np.arange(len(df_price)) * compact_ratio
bar_width = 0.30

# Create the figure and axis
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_box_aspect(1)

# Generate bar colors
colors = sns.color_palette("YlGnBu", len(df_price))

# Draw the bar chart
bars = ax.bar(
    x_positions,
    df_price['Price'],
    color=colors,
    alpha=0.9,
    edgecolor='black',
    linewidth=1.2,
    hatch='///',
    width=bar_width
)

# Use a logarithmic y-axis
ax.set_yscale('log')
ax.set_ylim(1e-4, 1)

# Set y-axis major ticks and labels
yticks = [1e-4, 1e-3, 1e-2, 1e-1, 1]
ylabels = [
    r'$10^{-4}$',
    r'$10^{-3}$',
    r'$10^{-2}$',
    r'$10^{-1}$',
    r'$10^{0}$'
]
ax.yaxis.set_major_locator(FixedLocator(yticks))
ax.yaxis.set_major_formatter(FixedFormatter(ylabels))
ax.yaxis.set_minor_locator(NullLocator())

# Set x-axis tick labels
ax.set_xticks(x_positions)
ax.set_xticklabels(
    df_price['Model'],
    fontsize=16,
    fontweight='bold',
    rotation=30,
    ha='right',
    rotation_mode='anchor'
)

# Remove internal grid lines
ax.grid(False)

# Set axis label and x-axis limits
ax.set_ylabel("Price (USD/paper)", fontsize=24, fontweight='bold')
ax.set_xlim(min(x_positions) - 0.20, max(x_positions) + 0.20)

# Format y-axis tick labels
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(18)

# Add numeric labels above each bar
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        height * 1.08,
        f'${height:.4f}',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )

# Draw black borders around the plotting area
for side in ['top', 'right', 'bottom', 'left']:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color('black')
    ax.spines[side].set_linewidth(2.0)

# Set tick appearance
ax.tick_params(axis='both', which='major', width=1.5, colors='black')
ax.tick_params(axis='y', which='minor', length=0)

plt.tight_layout()
plt.show()
