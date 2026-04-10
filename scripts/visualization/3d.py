import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from a relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "output_results.xlsx")

try:
    df = pd.read_excel(file_path, engine='openpyxl')
except Exception:
    df = pd.read_excel(file_path)

col_soc = 'SOC_Cleaned'
col_material = 'Cathode_Cleaned'

# Clean categorical labels
df[col_soc] = df[col_soc].astype(str).str.strip()
df[col_material] = df[col_material].astype(str).str.strip()

replace_dict = {
    '0': '0%',
    '1': '100%',
    '1.0': '100%',
    '0.0': '0%'
}
df[col_soc] = df[col_soc].replace(replace_dict)

# Compute the correlation matrix based on one-hot encoded categories
df_soc_encoded = pd.get_dummies(df[col_soc], prefix='', prefix_sep='')
df_mat_encoded = pd.get_dummies(df[col_material], prefix='', prefix_sep='')

df_all = pd.concat([df_mat_encoded, df_soc_encoded], axis=1)
full_corr = df_all.corr()

# Define the target categories for plotting
target_rows = ['NCM811', 'NCM622', 'NCM523', 'NCM111', 'LFP', 'Na-Ion']
target_cols = ['0%', '0%~50%', '50%~80%', '80%~100%', '100%', 'Overcharged']

# Figure size
FIG_SIZE = (6, 5.5)

# Extract and transpose the data so that SOC appears on the y-axis
try:
    raw_data = full_corr.loc[target_rows, target_cols]
    plot_data = raw_data.T
except KeyError as e:
    print(f"Error: missing category {e}")
    plot_data = None

if plot_data is not None:
    plt.figure(figsize=FIG_SIZE)

    # Create text annotations
    annot_data = plot_data.map(lambda x: f"{x:.2f}" if abs(x) >= 0.005 else "")

    # Draw heatmap
    ax = sns.heatmap(
        plot_data,
        annot=annot_data,
        fmt="",
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=1,
        linecolor='black',
        cbar=False
    )

    plt.title("")
    plt.xlabel("")
    plt.ylabel("")

    # Keep axis tick formatting consistent with the original figure
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=60)

    plt.tight_layout()
    plt.show()
