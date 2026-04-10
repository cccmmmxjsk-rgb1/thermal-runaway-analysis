import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 1. Read Excel file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "supplementary_data_trigger_temperature.xlsx")
df = pd.read_excel(file_path)

# =========================================================
# Step A: Data cleaning
# =========================================================
df.columns.values[0] = 'Material'
df.columns.values[1] = 'Trigger_Temperature'

# If Trigger_Temperature is a percentage string, convert it to numeric
if df['Trigger_Temperature'].dtype == 'object':
    df['Trigger_Temperature'] = df['Trigger_Temperature'].astype(str).str.rstrip('%').astype(float)

# Mapping of Mn content for each material
mn_map = {
    'LFP': 0,
    'NCM-111': 0.33, 'NCM111': 0.33,
    'NCM-523': 0.2, 'NCM523': 0.2,
    'NCM-622': 0.2, 'NCM622': 0.2,
    'NCM-811': 0.1, 'NCM811': 0.1,
    'NCA': 0
}

df['Mn_Content'] = df['Material'].astype(str).str.strip().map(mn_map)
df_plot = df.dropna(subset=['Mn_Content', 'Trigger_Temperature']).copy()

# =========================================================
# Step B: Calculate rank and jitter
# =========================================================
unique_mn_values = sorted(df_plot['Mn_Content'].unique())
rank_map = {val: i for i, val in enumerate(unique_mn_values)}
df_plot['X_Rank'] = df_plot['Mn_Content'].map(rank_map)

sorted_materials_for_legend = (
    df_plot.sort_values('Mn_Content')['Material'].unique().tolist()
)

region_width = 0.75
random_spread = (np.random.rand(len(df_plot)) - 0.5) * region_width
df_plot['X_Final'] = df_plot['X_Rank'] + random_spread

# =========================================================
# Step C: Calculate mean values and generate smoothed curve
# =========================================================
mean_data = df_plot.groupby('X_Rank')['Trigger_Temperature'].mean().sort_index()
x_nodes = mean_data.index.values
y_nodes = mean_data.values

if len(x_nodes) > 1:
    k_order = 3 if len(x_nodes) >= 4 else (len(x_nodes) - 1)
    spl = make_interp_spline(x_nodes, y_nodes, k=k_order)
    x_smooth = np.linspace(x_nodes.min(), x_nodes.max(), 300)
    y_smooth = spl(x_smooth)
else:
    x_smooth, y_smooth = x_nodes, y_nodes

# =========================================================
# Step D: Plot
# =========================================================
fig, ax = plt.subplots(figsize=(8, 8))
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Smoothed curve
ax.plot(x_smooth, y_smooth, color='black', linewidth=3, zorder=1, alpha=0.9)

# Scatter plot
sns.scatterplot(
    data=df_plot,
    x='X_Final',
    y='Trigger_Temperature',
    hue='Material',
    hue_order=sorted_materials_for_legend,
    palette={
        'LFP': '#1f77b4',
        'NCM-111': '#ff7f0e', 'NCM111': '#ff7f0e',
        'NCM-523': '#2ca02c', 'NCM523': '#2ca02c',
        'NCM-622': '#d62728', 'NCM622': '#d62728',
        'NCM-811': '#9467bd', 'NCM811': '#9467bd',
        'NCA': '#8c564b'
    }
    ,
    s=300,
    alpha=1.0,
    edgecolor=None,
    zorder=2,
    clip_on=False,
    legend=False,
    ax=ax
)

# Hollow-circle effect
path_collection = ax.collections[0]
original_colors = path_collection.get_facecolors()
path_collection.set_edgecolors(original_colors)
path_collection.set_facecolors('none')
path_collection.set_linewidth(2.5)

# =========================================================
# Step E: Axis formatting and styling
# =========================================================
ax.set_xticks(range(len(unique_mn_values)))
ax.set_xticklabels(unique_mn_values, fontweight='bold')
ax.tick_params(
    axis='both',
    which='major',
    labelsize=24,
    width=4,
    length=12,
    bottom=True,
    left=True,
    direction='out',
    pad=10
)

ax.set_xlabel("Mn Content (%)", fontsize=36, fontweight='bold', labelpad=20)
ax.set_ylabel("Trigger Temperature (°C)", fontsize=36, fontweight='bold', labelpad=20)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(4)
    spine.set_color('black')

ax.set_ylim(0, 450)
ax.set_xlim(-0.6, len(unique_mn_values) - 0.4)

# Make the plotting area square
ax.set_box_aspect(1)

# Hide the first y-axis tick line
y_ticks = ax.yaxis.get_major_ticks()
if len(y_ticks) > 0:
    y_ticks[0].tick1line.set_visible(False)

plt.tight_layout()
plt.savefig('supp_fig_s3_mn_content_vs_trigger_temperature.png', dpi=300, bbox_inches='tight')
plt.show()
