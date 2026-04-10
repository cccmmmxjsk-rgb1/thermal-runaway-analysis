import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Locate the input data file relative to the script
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "correlation_analysis_data.xlsx")

col_phenomenon = 'observed_results.phenomenon'
col_material = 'Cathode_Cleaned'

target_materials = ['NCM-General', 'NCM811', 'LFP', 'NCM-622', 'Na-Ion', 'NCA', 'NCM111']
target_phenomena = ['No Fire', 'Smoking', 'Jetting', 'Fire', 'Explosion']

colors = ["#b2182b", "#f7f7f7", "#053061"]
my_red_blue = LinearSegmentedColormap.from_list("custom_rd_bu", colors)

try:
    df = pd.read_excel(file_path)
except ImportError:
    print("\n[Missing Dependency]")
    print("The 'openpyxl' library is required to read .xlsx files.")
    print("Please run: pip install openpyxl\n")
    exit()
except FileNotFoundError:
    print("\n[File Not Found]")
    print(f"Cannot find the data file at: {file_path}")
    print("Please ensure '4d.xlsx' is located in the 'data' directory.\n")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# Standardize string formatting
df[col_phenomenon] = df[col_phenomenon].astype(str).str.strip()
df[col_material] = df[col_material].astype(str).str.strip()

replace_dict = {
    '0': '0%',
    '1': '100%',
    '1.0': '100%',
    '0.0': '0%'
}
df[col_phenomenon] = df[col_phenomenon].replace(replace_dict)

# One-hot encoding for correlation analysis
df_phenom_encoded = pd.get_dummies(df[col_phenomenon], prefix='', prefix_sep='')
df_mat_encoded = pd.get_dummies(df[col_material], prefix='', prefix_sep='')

df_all = pd.concat([df_mat_encoded, df_phenom_encoded], axis=1)
full_corr = df_all.corr()

# Select variables available in the dataset
available_materials = [m for m in target_materials if m in full_corr.index]
available_phenomena = [p for p in target_phenomena if p in full_corr.columns]

if not available_materials or not available_phenomena:
    print("Warning: Target materials or phenomena are missing from the dataset.")
else:
    plot_data = full_corr.loc[available_materials, available_phenomena].T

    plt.figure(figsize=(6, 5))

    ax = sns.heatmap(
        plot_data,
        annot=True,
        fmt='.2f',
        cmap=my_red_blue,
        vmin=-0.15,
        vmax=0.15,
        center=0,
        cbar=False,
        linewidths=0.5,
        linecolor='white',
        square=True,
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
    plt.savefig('Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
