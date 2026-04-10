import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import ScalarFormatter

# ==========================================
# 1. Data Preparation
# ==========================================

# Define relative path to the dataset
# (Note: Ensure the data file is placed in a 'data' folder within the repository)
file_path = "../data/processing_time.xlsx"

try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# ==========================================
# 2. Core Plotting Logic
# ==========================================

# Set font configurations
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Clean column names
df.columns = df.columns.str.strip()
target_models = df.columns.tolist()

# Sort models based on median values in descending order
medians = df[target_models].median().sort_values(ascending=False)
sorted_models = medians.index.tolist()

# Define color palette
palette = sns.color_palette("viridis", len(sorted_models))

# Create canvas
fig, ax = plt.subplots(figsize=(9, 4))

# Layout parameters for the half-violin (raincloud) plot
shift_box = -0.15
shift_scatter = 0.18
shift_violin = 0.05
width_box = 0.2
scale_violin = 0.35

# Set Y-axis view limits
y_view_min = 1
y_view_max = 1000

for i, model in enumerate(sorted_models):
    y_data = df[model].dropna().values

    # Filter out values <= 0 (Log axis cannot display 0 or negative numbers)
    y_data = y_data[y_data > 0]

    if len(y_data) == 0:
        continue

    base_color = palette[i]
    fill_color = list(base_color)
    if len(fill_color) == 3:
        fill_color.append(0.4)
    else:
        fill_color = list(base_color[:3]) + [0.4]

    x_center = i

    # A. Boxplot
    ax.boxplot([y_data], positions=[x_center + shift_box], widths=width_box,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=fill_color, edgecolor=base_color, linewidth=1.5),
               medianprops=dict(color='white', linewidth=1.5),
               whiskerprops=dict(color=base_color, linewidth=1.5),
               capprops=dict(color=base_color, linewidth=1.5))

    # B. Scatter plot with jitter
    jitter = np.random.uniform(-0.05, 0.25, size=len(y_data))
    ax.scatter(np.full_like(y_data, x_center + shift_scatter) + jitter, y_data,
               facecolors='none', edgecolors=base_color, s=20, linewidth=1, zorder=2, alpha=0.8)

    # C. Density line (KDE) - Special handling for Log axis
    if len(y_data) > 1:
        try:
            kde = stats.gaussian_kde(y_data)

            # Generate grid using logspace to ensure smooth KDE curves on a logarithmic axis
            y_grid = np.logspace(np.log10(max(y_data.min(), 0.1)), np.log10(y_data.max()), 200)

            kde_vals = kde(y_grid)
            kde_vals = kde_vals / kde_vals.max() * scale_violin

            ax.plot(x_center + shift_violin + kde_vals, y_grid, color=base_color, linewidth=2, alpha=0.9)
        except Exception:
            pass

# ==============================================================================
# 3. Formatting and Log Axis Configuration
# ==============================================================================

ax.set_xticks(range(len(sorted_models)))
ax.set_xticklabels(sorted_models, rotation=15, ha='center', fontsize=11, fontweight='bold')

# Set Y-axis to logarithmic scale
ax.set_yscale('log')
ax.set_ylim(y_view_min, y_view_max)

# Manually set major tick positions
major_ticks = [1, 10, 100, 1000]
ax.set_yticks(major_ticks)

# Manually set tick labels to standard notation instead of scientific notation (e.g., 10^1)
ax.set_yticklabels([str(t) for t in major_ticks], fontsize=11, fontweight='bold')

# Remove minor ticks for a cleaner appearance
ax.yaxis.set_minor_locator(plt.NullLocator())

# Labels and spines
ax.set_ylabel("Processing Time (s)", fontsize=13, fontweight='bold')

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

# Add grid lines for major ticks
ax.yaxis.grid(True, linestyle='-', color='#87CEEB', alpha=0.8, zorder=0)
ax.xaxis.grid(False)

plt.tight_layout()
plt.savefig('2a.png', dpi=300, bbox_inches='tight')
plt.show()