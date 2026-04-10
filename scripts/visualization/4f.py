import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "gas_material_correlation_data.xlsx")

gas_columns = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']
target_materials = ['NCM-General', 'NCM811', 'LFP', 'NCM622', 'Na-Ion', 'NCA', 'NCM111']

reference_colors = ['#003366', '#336699', '#85A3C2', '#ECEFF1', '#E09999', '#CC3333', '#990000']
custom_cmap = LinearSegmentedColormap.from_list('reference_style', reference_colors)

try:
    df = pd.read_excel(file_path)

    # Filter target materials
    df_filtered = df[df['Cathode_Cleaned'].isin(target_materials)]

    if df_filtered.empty:
        print("Warning: No valid data were found.")
    else:
        # Prepare data for correlation analysis
        df_dummies = pd.get_dummies(df_filtered['Cathode_Cleaned'])

        df_analysis = pd.concat(
            [
                df_filtered[gas_columns].reset_index(drop=True),
                df_dummies.reset_index(drop=True)
            ],
            axis=1
        )

        corr_matrix = df_analysis.corr()

        available_materials = [m for m in target_materials if m in corr_matrix.index]
        plot_data = corr_matrix.loc[available_materials, gas_columns].T

        plt.figure(figsize=(6, 5))

        ax = sns.heatmap(
            plot_data,
            annot=True,
            cmap=custom_cmap,
            vmin=-0.3,
            vmax=0.3,
            center=0,
            fmt='.2f',
            linewidths=0.5,
            linecolor='white',
            cbar=False,
            square=True,
            annot_kws={"size": 10, "color": "black"}
        )

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')

        plt.xlabel("")
        plt.ylabel("")

        plt.xticks(fontsize=11, rotation=45, color='black')
        plt.yticks(fontsize=11, rotation=60, color='black')

        plt.tight_layout()
        plt.show()

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"An error occurred: {e}")
