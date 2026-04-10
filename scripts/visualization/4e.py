import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "mass_material_correlation_data.xlsx")

target_materials = ['NCM-General', 'NCM811', 'LFP', 'NCM-622', 'Na-Ion', 'NCA', 'NCM111']
mass_order = ['<10%', '10%~20%', '20%~30%', '30%~40%', '40%~60%', '>60%']

reference_colors = ['#003366', '#336699', '#85A3C2', '#ECEFF1', '#E09999', '#CC3333', '#990000']
custom_cmap = LinearSegmentedColormap.from_list('reference_style', reference_colors)

try:
    df = pd.read_excel(file_path)

    # Filter target materials and mass categories
    df_filtered = df[df['Cathode_Cleaned'].isin(target_materials)].copy()
    df_filtered = df_filtered[df_filtered['mass'].isin(mass_order)]

    if df_filtered.empty:
        print("Warning: No valid data were found.")
    else:
        # One-hot encoding for correlation analysis
        mass_dummies = pd.get_dummies(df_filtered['mass'])
        mat_dummies = pd.get_dummies(df_filtered['Cathode_Cleaned'])

        full_corr = pd.concat([mass_dummies, mat_dummies], axis=1).corr()

        existing_mass = [m for m in mass_order if m in mass_dummies.columns]
        existing_mats = [m for m in target_materials if m in mat_dummies.columns]

        plot_data = full_corr.loc[existing_mass, existing_mats]

        plt.figure(figsize=(6, 5))

        ax = sns.heatmap(
            plot_data,
            annot=True,
            cmap=custom_cmap,
            vmin=-0.5,
            vmax=0.5,
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
