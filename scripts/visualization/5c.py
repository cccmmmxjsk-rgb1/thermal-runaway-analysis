import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# =========================================================
# Global Font Settings (Journal Standard)
# =========================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# Path Configuration for Supplementary Material
# =========================================================
# Relative path to the dataset
DATA_PATH = r'../data/supplementary_data_trigger_temperature.xlsx'
# Save figures to the current working directory
SAVE_DIR = r'.'

# =========================================================
# Step A: Data Loading & Cleaning
# =========================================================
df = pd.read_excel(DATA_PATH)

# Standardize column names
df.columns.values[0] = 'Material'
df.columns.values[1] = 'Trigger_Temperature'

# Clean percentage strings if they exist
if df['Trigger_Temperature'].dtype == 'object':
    df['Trigger_Temperature'] = df['Trigger_Temperature'].astype(str).str.rstrip('%').astype(float)

# Exclude LFP material
df = df[~df['Material'].astype(str).str.contains('LFP')].copy()

# Filter out extreme outliers (Trigger_Temperature > 600 °C)
df = df[df['Trigger_Temperature'] <= 600].copy()

# Element content mapping based on material nomenclature
ni_map = {'NCM-111': 0.33, 'NCM111': 0.33, 'NCM-523': 0.5, 'NCM523': 0.5, 'NCM-622': 0.6, 'NCM622': 0.6, 'NCM-811': 0.8, 'NCM811': 0.8, 'NCA': 0.85}
co_map = {'NCM-111': 0.33, 'NCM111': 0.33, 'NCM-523': 0.20, 'NCM523': 0.20, 'NCM-622': 0.20, 'NCM622': 0.20, 'NCM-811': 0.10, 'NCM811': 0.10, 'NCA': 0.10}
mn_map = {'NCM-111': 0.33, 'NCM111': 0.33, 'NCM-523': 0.3, 'NCM523': 0.3, 'NCM-622': 0.2, 'NCM622': 0.2, 'NCM-811': 0.1, 'NCM811': 0.1, 'NCA': 0}

df['Ni'] = df['Material'].astype(str).str.strip().map(ni_map)
df['Co'] = df['Material'].astype(str).str.strip().map(co_map)
df['Mn'] = df['Material'].astype(str).str.strip().map(mn_map)

# Extract sorted unique values for categorical plotting
ni_vals = sorted(df['Ni'].dropna().unique())
co_vals = sorted(df['Co'].dropna().unique())
mn_vals = sorted(df['Mn'].dropna().unique())

# =========================================================
# Step B: Layout & X-axis Coordinate Calculation
# =========================================================
spacing = 2
x_ni = np.arange(len(ni_vals))
x_co = np.arange(len(co_vals)) + x_ni[-1] + spacing
x_mn = np.arange(len(mn_vals)) + x_co[-1] + spacing

fig, ax = plt.subplots(figsize=(15, 4.1))

# Raincloud plot parameter settings
shift_box = -0.15
shift_scatter = 0.15
shift_violin = 0.0
width_box = 0.25
scale_violin = 0.3

# =========================================================
# Step C: Core Plotting Function (Raincloud Plot)
# =========================================================
def plot_element_region(val_list, col_name, x_positions, cmap_name):
    # Generate sequential colormap for the elements
    colors = sns.color_palette(cmap_name, n_colors=len(val_list) + 2)[2:]

    for i, val in enumerate(val_list):
        y_data = df[df[col_name] == val]['Trigger_Temperature'].dropna().values
        if len(y_data) == 0:
            continue

        base_color = colors[i]
        fill_color = list(base_color) + [0.4]  # Add alpha for transparency
        x_center = x_positions[i]

        median_val = np.median(y_data)
        std_val = np.std(y_data)

        # 1. Boxplot (Left side)
        ax.boxplot(
            [y_data], positions=[x_center + shift_box], widths=width_box,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=fill_color, edgecolor=base_color, linewidth=2),
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color=base_color, linewidth=2),
            capprops=dict(color=base_color, linewidth=2)
        )

        # 2. Median marker
        ax.plot(x_center + shift_box, median_val, marker='D', color=base_color,
                markersize=5, linestyle='None', markeredgecolor=base_color, zorder=3)

        # 3. Scatter plot with jitter (Right side)
        jitter = np.random.uniform(-0.06, 0.06, size=len(y_data))
        ax.scatter(np.full_like(y_data, x_center + shift_scatter) + jitter, y_data,
                   facecolors='none', edgecolors=base_color, s=20, linewidth=1.2, zorder=2)

        # 4. Half-violin (KDE curve)
        if len(y_data) > 1:
            try:
                kde = stats.gaussian_kde(y_data)
                y_grid = np.linspace(max(0, y_data.min() - 20), min(600, y_data.max() + 20), 200)
                kde_vals = kde(y_grid)
                if kde_vals.max() > 0:
                    kde_vals = kde_vals / kde_vals.max() * scale_violin
                ax.plot(x_center + shift_violin + kde_vals, y_grid, color=base_color, linewidth=2.5)
            except Exception:
                pass

        # 5. Statistical annotation (Median ± Std)
        text_str = f"{median_val:.1f}±{std_val:.1f}"
        text_y_pos = min(np.max(y_data) + 25, 520)

        ax.text(
            x_center, text_y_pos, text_str,
            ha='center', va='bottom',
            fontsize=13, color='black', fontweight='bold', fontfamily='sans-serif'
        )

# Plot each element group
plot_element_region(ni_vals, 'Ni', x_ni, "Reds")
plot_element_region(co_vals, 'Co', x_co, "Greens")
plot_element_region(mn_vals, 'Mn', x_mn, "Blues")

# =========================================================
# Step D: Axis & Aesthetics Formatting
# =========================================================
all_x = list(x_ni) + list(x_co) + list(x_mn)
all_labels = [f"{v:.2g}" for v in ni_vals] + [f"{v:.2g}" for v in co_vals] + [f"{v:.2g}" for v in mn_vals]

ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=16, fontweight='bold', color='black')

ax.set_ylabel("Trigger Temperature (°C)", fontsize=18, fontweight='bold', color='black')
ax.set_ylim(0, 600)
ax.set_yticks([0, 100, 200, 300, 400, 500, 600])

for label in ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_fontweight('bold')

# Group separators
sep1 = (x_ni[-1] + x_co[0]) / 2
sep2 = (x_co[-1] + x_mn[0]) / 2
ax.axvline(sep1, color='grey', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(sep2, color='grey', linestyle='--', linewidth=2, alpha=0.7)

# Group headers
ax.text(np.mean(x_ni), 1.03, "Ni Content", transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=18, fontweight='bold', color='#cb181d')
ax.text(np.mean(x_co), 1.03, "Co Content", transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=18, fontweight='bold', color='#238b45')
ax.text(np.mean(x_mn), 1.03, "Mn Content", transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=18, fontweight='bold', color='#2171b5')

# Spine formatting
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('black')

# Grid settings
ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.7, linewidth=1)
ax.set_axisbelow(True)

plt.tight_layout()

# Save the figure silently
output_filename = os.path.join(SAVE_DIR, 'Figure_Raincloud_Elements.png')
plt.savefig(output_filename, dpi=500, bbox_inches='tight')
plt.close()
