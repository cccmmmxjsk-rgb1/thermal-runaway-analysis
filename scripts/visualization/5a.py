import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "trigger_temperature_data.xlsx"
)

value_col_name = 'observed_results.T_trigger'

target_materials = [
    'NCM111', 'NCM811', 'LFP', 'NCM622', 'Na-Ion', 'NCA', 'NCM-General', 'NCM523'
]

y_min, y_max = 0, 500

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Read and validate the dataset
try:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    if value_col_name not in df.columns:
        print(f"\n[Error] Column not found in the Excel file: '{value_col_name}'")
        raise KeyError(f"Column '{value_col_name}' not found")

except FileNotFoundError:
    print(f"Error: File not found. Please check the file path: {file_path}")
    exit()

# Filter target materials
df_filtered = df[df['Cathode_Cleaned'].isin(target_materials)].copy()

# Order materials by median trigger temperature
models = df_filtered.groupby('Cathode_Cleaned')[value_col_name].median().sort_values(ascending=False).index

palette = sns.color_palette("viridis_r", len(models))

fig, ax = plt.subplots(figsize=(15, 4))

shift_box = -0.2
shift_scatter = 0.2
shift_violin = 0.05
width_box = 0.25
scale_violin = 0.4

for i, model in enumerate(models):
    y_data = df_filtered[df_filtered['Cathode_Cleaned'] == model][value_col_name].values
    if len(y_data) == 0:
        continue

    base_color = palette[i]
    fill_color = list(base_color) + [0.4]
    x_center = i

    # Box plot
    ax.boxplot(
        [y_data],
        positions=[x_center + shift_box],
        widths=width_box,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        boxprops=dict(facecolor=fill_color, edgecolor=base_color, linewidth=2),
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color=base_color, linewidth=2),
        capprops=dict(color=base_color, linewidth=2)
    )

    # Scatter plot
    jitter = np.random.uniform(-0.08, 0.08, size=len(y_data))
    ax.scatter(
        np.full_like(y_data, x_center + shift_scatter) + jitter,
        y_data,
        facecolors='none',
        edgecolors=base_color,
        s=25,
        linewidth=1,
        zorder=2
    )

    # Density curve
    if len(y_data) > 1:
        try:
            kde = stats.gaussian_kde(y_data)
            y_grid = np.linspace(y_data.min() - 10, y_data.max() + 10, 200)
            kde_vals = kde(y_grid)
            if kde_vals.max() > 0:
                kde_vals = kde_vals / kde_vals.max() * scale_violin
            ax.plot(
                x_center + shift_violin + kde_vals,
                y_grid,
                color=base_color,
                linewidth=2.5
            )
        except Exception:
            pass

ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=0, ha='center', fontsize=18, color='black', fontweight='bold')
ax.set_ylabel("Trigger Temperature (°C)", fontsize=18, fontweight='bold', color='black')
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(y_min, y_max)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

ax.yaxis.grid(True, linestyle='-', color='#e0e0e0', alpha=0.8, zorder=0)
ax.xaxis.grid(False)

plt.tight_layout()
plt.savefig('Cathode_Raincloud_Plot_Fig2c_Style.png', dpi=300, bbox_inches='tight')
plt.show()
