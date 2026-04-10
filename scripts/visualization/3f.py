import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data from a relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "output_results.xlsx")
df = pd.read_excel(file_path)

col_material = 'Cathode_Cleaned'
col_heating = 'Heating_Cleaned'

# Clean categorical labels
df[col_material] = df[col_material].astype(str).str.strip()
df[col_heating] = df[col_heating].astype(str).str.strip()

# Compute the correlation matrix with prefixes to avoid duplicated column names
df_mat_encoded = pd.get_dummies(df[col_material], prefix='Mat', prefix_sep='_')
df_heat_encoded = pd.get_dummies(df[col_heating], prefix='Heat', prefix_sep='_')

df_all = pd.concat([df_mat_encoded, df_heat_encoded], axis=1)
full_corr = df_all.corr()

# Define target categories
target_rows = ['Global', 'Side', 'Bottom', 'Wrap', 'Surface', 'Induced', 'others']
target_cols = ['NCM-General', 'NCM811', 'NCM622', 'NCM523', 'NCM111', 'LFP', 'NCA']

# Build prefixed labels for indexing
lookup_rows = [f"Heat_{x}" for x in target_rows]
lookup_cols = [f"Mat_{x}" for x in target_cols]

# Extract plotting data and restore original labels
plot_data = full_corr.reindex(index=lookup_rows, columns=lookup_cols).fillna(0)
plot_data.index = target_rows
plot_data.columns = target_cols

# Define colormap
colors = ["#FFFFFF", "#F0F0F0", "#CDBEB8", "#8B7D7B"]
cmap_custom = mcolors.LinearSegmentedColormap.from_list("CustomBrown", colors, N=100)

# Draw heatmap
if plot_data is not None:
    plt.figure(figsize=(6, 5))

    annot_data = plot_data.map(lambda x: f"{x:.2f}" if abs(x) >= 0.02 else "")

    ax = sns.heatmap(
        plot_data,
        annot=annot_data,
        fmt="",
        cmap=cmap_custom,
        vmin=0,
        vmax=plot_data.max().max(),
        cbar=False,
        square=True,
        linewidths=0,
        annot_kws={"size": 10, "color": "black"}
    )

    plt.title("")
    plt.xlabel("")
    plt.ylabel("")

    plt.xticks(fontsize=11, rotation=45, color='black')
    plt.yticks(fontsize=11, rotation=60, color='black')

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.show()
