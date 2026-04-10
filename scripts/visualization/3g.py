import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data from a relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "output_results.xlsx")
df = pd.read_excel(file_path)

col_x = 'Cathode_Cleaned'
col_y = 'Form_Cleaned'

# Clean categorical labels
try:
    df[col_x] = df[col_x].astype(str).str.strip()
    df[col_y] = df[col_y].astype(str).str.strip()
except KeyError as e:
    print(f"Error: column {e} was not found in the Excel file.")

# Compute the correlation matrix with prefixes to avoid duplicated column names
df_x_encoded = pd.get_dummies(df[col_x], prefix='Mat', prefix_sep='_')
df_y_encoded = pd.get_dummies(df[col_y], prefix='Form', prefix_sep='_')

df_all = pd.concat([df_x_encoded, df_y_encoded], axis=1)
full_corr = df_all.corr()

# Define target categories
target_rows = ['Cylindrical', 'Pouch', 'Prismatic', 'Coin Cell', 'Pellet', 'others']
target_cols = ['NCM-General', 'NCM811', 'NCM622', 'NCM523', 'NCM111', 'LFP', 'NCA']

# Build prefixed labels for indexing
target_rows_prefixed = [f'Form_{x}' for x in target_rows]
target_cols_prefixed = [f'Mat_{x}' for x in target_cols]

# Extract plotting data and restore original labels
plot_data = full_corr.reindex(index=target_rows_prefixed, columns=target_cols_prefixed).fillna(0)
plot_data.index = target_rows
plot_data.columns = target_cols

# Define colormap
colors = ["#FFFFFF", "#F9FBE7", "#E6EE9C", "#DCE775"]
cmap_custom = mcolors.LinearSegmentedColormap.from_list("CustomYellowGreen", colors, N=100)

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
