import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mass_gas_correlation_data.xlsx")
gas_columns = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']
mass_column = 'mass'

mass_order = [
    '<10%',
    '10%~20%',
    '20%~30%',
    '30%~40%',
    '40%~60%',
    '>60%'
]

cmap_style = sns.light_palette("#002b41", as_cmap=True)

try:
    df = pd.read_excel(file_path)

    # Convert gas concentration columns to numeric values
    for col in gas_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if mass_column not in df.columns:
        raise ValueError(f"Column '{mass_column}' was not found in the dataset.")

    # One-hot encoding for correlation analysis
    df_dummies = pd.get_dummies(df[mass_column])
    df_analysis = pd.concat([df[gas_columns], df_dummies], axis=1)
    full_corr = df_analysis.corr()

    available_mass = [m for m in mass_order if m in full_corr.index]
    plot_data = full_corr.loc[available_mass, gas_columns]

    plt.figure(figsize=(6, 5))

    ax = sns.heatmap(
        plot_data,
        annot=True,
        cmap=cmap_style,
        vmin=-0.2,
        vmax=0.3,
        center=None,
        fmt='.2f',
        linewidths=0.5,
        linecolor='lightgrey',
        cbar=False,
        square=True,
        annot_kws={"size": 10, "color": "black"}
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.xticks(fontsize=11, rotation=45, color='black')
    plt.yticks(fontsize=11, rotation=60, color='black')

    plt.tight_layout()
    plt.show()

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"An error occurred: {e}")
