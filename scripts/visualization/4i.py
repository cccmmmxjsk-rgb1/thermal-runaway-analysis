import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "phenomenon_gas_correlation_data.xlsx")

gas_columns = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']
phenomenon_column = 'observed_results.phenomenon'

phenomenon_order = [
    'No Fire',
    'Smoking',
    'Jetting',
    'Fire',
    'Explosion'
]

cmap_style = sns.light_palette("red", as_cmap=True)

try:
    df = pd.read_excel(file_path)

    # Convert gas columns to numeric values
    for col in gas_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if phenomenon_column not in df.columns:
        raise ValueError(f"Column '{phenomenon_column}' was not found in the dataset.")

    # One-hot encoding for phenomenon categories
    df_dummies = pd.get_dummies(df[phenomenon_column])

    df_analysis = pd.concat([df[gas_columns], df_dummies], axis=1)
    full_corr = df_analysis.corr()

    available_phenomena = [p for p in phenomenon_order if p in full_corr.index]
    plot_data = full_corr.loc[available_phenomena, gas_columns]

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
