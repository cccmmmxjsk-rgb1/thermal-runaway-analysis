import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Global font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# Load data from a relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "evaluation_summary_data.xlsx")
df = pd.read_excel(file_path)

# Rank models by the mean Total_Score in descending order
models = df.groupby('Model')['Total_Score'].mean().sort_values(ascending=False).index

# Count zero-score cases for each model
zero_count_map = df[df['Total_Score'] == 0].groupby('Model').size().to_dict()

# Initialize figure and color palette
fig, ax = plt.subplots(figsize=(16, 5))
palette = sns.color_palette("viridis_r", len(models))

# Layout parameters
shift_box = -0.15
shift_scatter = 0.15
shift_violin = 0.0
width_box = 0.25
scale_violin = 0.35

# Plot each model
for i, model in enumerate(models):
    y_data = df[df['Model'] == model]['Total_Score'].values

    if len(y_data) == 0:
        continue

    # Color settings
    base_color = palette[i]
    fill_color = list(base_color) + [0.4]
    x_center = i

    # Compute summary statistics
    mean_val = np.mean(y_data)
    std_val = np.std(y_data)

    # Box plot with mean marker
    ax.boxplot(
        [y_data],
        positions=[x_center + shift_box],
        widths=width_box,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=fill_color, edgecolor=base_color, linewidth=2.5),
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(color=base_color, linewidth=2.5),
        capprops=dict(color=base_color, linewidth=2.5)
    )

    ax.plot(
        x_center + shift_box,
        mean_val,
        marker='D',
        color=base_color,
        markersize=6,
        linestyle='None',
        markeredgecolor=base_color,
        zorder=3
    )

    # Scatter plot
    jitter = np.random.uniform(-0.06, 0.06, size=len(y_data))
    ax.scatter(
        np.full_like(y_data, x_center + shift_scatter) + jitter,
        y_data,
        facecolors='none',
        edgecolors=base_color,
        s=25,
        linewidth=1.5,
        zorder=2
    )

    # Density curve
    if len(y_data) > 1:
        try:
            kde = stats.gaussian_kde(y_data)
            y_grid = np.linspace(y_data.min(), y_data.max() + 5, 200)
            kde_vals = kde(y_grid)
            if kde_vals.max() > 0:
                kde_vals = kde_vals / kde_vals.max() * scale_violin
            ax.plot(
                x_center + shift_violin + kde_vals,
                y_grid,
                color=base_color,
                linewidth=3
            )
        except Exception:
            pass

    # Text annotation: Mean ± SD
    text_str = f"{mean_val:.1f}±{std_val:.1f}"

    if model == 'Qwen-Turbo':
        text_y = 6
        va = 'bottom'
    elif mean_val < 70:
        text_y = 12
        va = 'bottom'
    else:
        text_y = 37
        va = 'top'

    ax.text(
        x_center,
        text_y,
        text_str,
        ha='center',
        va=va,
        fontsize=15,
        color='black',
        fontweight='bold',
        fontfamily='sans-serif'
    )

# Format x-axis labels
name_map = {
    'deepseekr1': 'DeepSeek\n-R1',
    'gpt5': 'GPT-5',
    'deepseek-v3': 'DeepSeek\n-V3',
    'claude4': 'Claude 4',
    'qwen-plus': 'Qwen\nPlus',
    'gemini2.5flash': 'Gemini\nFlash 2.0',
    'GPT-4o': 'GPT-4o',
}
final_labels = [name_map.get(m, m) for m in models]

ax.set_xticks(range(len(models)))
ax.set_xticklabels(
    final_labels,
    rotation=0,
    ha='center',
    fontsize=16,
    color='black',
    fontweight='bold',
    fontfamily='sans-serif'
)

# Add red labels for unidentified cases below the x-axis
for i, model in enumerate(models):
    zero_count = zero_count_map.get(model, 0)
    if zero_count > 0:
        ax.text(
            i,
            -0.12,
            f"({zero_count} Unidentified)",
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=12,
            color='red',
            fontweight='bold',
            fontfamily='sans-serif'
        )

ax.set_ylabel(
    "Score",
    fontsize=18,
    fontweight='bold',
    color='black',
    fontfamily='sans-serif'
)

# Format y-axis ticks
for label in ax.get_yticklabels():
    label.set_fontfamily('sans-serif')
    label.set_fontsize(16)
    label.set_fontweight('bold')

ax.set_ylim(0, 105)
ax.set_yticks([0, 20, 40, 60, 80, 100])

# Draw borders
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('black')

# Add grid lines
ax.yaxis.grid(True, linestyle='-', color='skyblue', alpha=1.0, linewidth=1)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('raincloud_final.png', dpi=500, bbox_inches='tight')
plt.show()
