import os
import re
import platform

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Input path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "..", "data", "tsne_analysis_data.xlsx")

# Target cathode groups
TARGET_CATHODES_LIST = [
    ['NCM811', 'NCM-622', 'NCM-General', 'NCM-111', 'NCM523', 'LFP'],
]

# Variables highlighted in the plot
HUE_LIST = ['cathode']

# Column mapping
COLUMN_MAPPING = {
    'cathode': 'cathode',
    'capacity': 'capacity',
    'tr_occurred': 'outcome',
    'trigger_method': 'trigger_method',
    'cell_format': 'cell_format',
    'electrolyte': 'electrolyte',
    'separator': 'separator',
    'atmosphere': 'atmosphere',
    'pressure': 'pressure',
    'safety_design': 'safety_design',
    'phenomenon': 'phenomenon',
    'gas_data': 'gas_data',
    'heating_side': 'heating_side',
    't_trigger': 't_trigger',
    't_max': 't_max'
}

# Region definitions for confidence ellipses
ELLIPSES_CONFIG = {
    "Region 1 (Left Island)": {
        "type": "rect",
        "x_range": [-22, -10],
        "y_range": [-18, -8],
        "n_std": 2.5,
        "color": "#455A64"
    },
    "Region 2 (Top Left)": {
        "type": "rect",
        "x_range": [-20, -5],
        "y_range": [5, 20],
        "n_std": 2.5,
        "color": "#B7950B"
    },
    "Region 3 (Bottom Center)": {
        "type": "rect",
        "x_range": [-2, 12],
        "y_range": [-16, 0],
        "n_std": 2,
        "color": "#A04000"
    },
    "Region 4 (Right Edge)": {
        "type": "rect",
        "x_range": [13, 22],
        "y_range": [-12, 6],
        "n_std": 3.5,
        "color": "#6D4C41"
    },
    "Region 5 (Core Polygon)": {
        "type": "poly",
        "points": [
            (-12, -8), (-12, 0), (-8, 8), (-5, 12), (5, 16),
            (12, 14), (12, 5), (5, -2), (0, -5), (-5, -10)
        ],
        "n_std": 2.5,
        "color": "#2E4053"
    }
}

# Custom annotation labels
LABELS_CONFIG = [
    {"text": "Safe/Suppressed (No Fire)\nEster Carbonates & Cooling", "x": -16.5, "y": -12.5, "size": 24, "color": "#2E4053"},
    {"text": "~193°C Trigger (Stable)\nEC:EMC (3:7) System", "x": -13, "y": 11.5, "size": 24, "color": "#2E4053"},
    {"text": "~150°C Low Trigger (Explosive)\nStandard EC/DEC/DMC", "x": 6, "y": -8.5, "size": 24, "color": "#2E4053"},
    {"text": "~213°C High Trigger (Fire)\nEC:DEC:DMC / Commercial", "x": 14, "y": -2, "size": 24, "color": "#2E4053"},
    {"text": "~211°C Trigger (LFP Band)\nLiPF6 Mixed / Solid-State Trials", "x": -1, "y": 1.5, "size": 24, "color": "#2E4053"},
]

# Weight assigned to numeric variables in the fused feature space
NUMERIC_WEIGHT = 4.0


def set_english_pub_style():
    """Apply a publication-style figure format."""
    font_family = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_family
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['legend.title_fontsize'] = 20

    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['svg.fonttype'] = 'none'

    print(f"Configured English publication style ({platform.system()})")


set_english_pub_style()


def clean_numeric(val):
    """Extract the first numeric value from a mixed-format field."""
    if pd.isna(val) or str(val).lower() in ['unknown', 'nan', 'none']:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    match = re.search(r"(\d+(\.\d+)?)", str(val))
    return float(match.group(1)) if match else 0.0


def load_data_all(file_path, mapping):
    """Load the dataset and standardize column names."""
    print(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        return None

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    df = df.rename(columns=mapping)

    for internal_name in mapping.values():
        if internal_name not in df.columns:
            df[internal_name] = np.nan

    for col in ['t_trigger', 't_max', 'capacity']:
        df[f'{col}_clean'] = df[col].apply(clean_numeric)

    exclude_cols = ['t_trigger', 't_max', 't_trigger_clean', 't_max_clean', 'capacity_clean']
    text_cols = [c for c in df.columns if c not in exclude_cols]
    for col in text_cols:
        df[col] = df[col].fillna('unknown').astype(str)

    return df


def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to transformer token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_bert_embeddings(text_list):
    """Generate sentence embeddings using a transformer model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "sentence-transformers/all-mpnet-base-v2"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    embeddings = []
    model.eval()

    for i in range(0, len(text_list), 32):
        batch = text_list[i:i + 32]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(**inputs)
            embeddings.append(mean_pooling(output, inputs['attention_mask']).cpu().numpy())

    return np.vstack(embeddings)


def draw_confidence_ellipse(x, y, ax, n_std=2.5, edgecolor='#2E4053'):
    """Draw a confidence ellipse for a selected region."""
    if len(x) < 3:
        return

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))

    ell_fill = patches.Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        color=edgecolor,
        alpha=0.08,
        zorder=1
    )
    ax.add_patch(ell_fill)

    ell_edge = patches.Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        edgecolor=edgecolor,
        facecolor='none',
        linestyle=(0, (5, 5)),
        linewidth=2.2,
        alpha=0.9,
        zorder=100
    )
    ax.add_patch(ell_edge)


def plot_tsne(df, x_col, y_col, hue_col, save_path, cathode_group_name):
    """Plot the t-SNE projection with custom annotations and regions."""
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.set_style("white")
    set_english_pub_style()
    ax.set_box_aspect(1)

    cathode_order_map = {
        'LFP': 0, 'NCM-111': 1, 'NCM523': 2,
        'NCM-General': 3, 'NCM-622': 4, 'NCM811': 5
    }

    plot_df = df.copy()
    plot_df['outcome'] = pd.to_numeric(plot_df['outcome'], errors='coerce').fillna(0).astype(int)
    tr_markers = {1: 'o', 0: 'X'}

    if hue_col == 'cathode':
        plot_df['cathode_rank'] = plot_df['cathode'].map(cathode_order_map)
        plot_df = plot_df.dropna(subset=['cathode_rank'])
        current_hue_col = 'cathode_rank'
        is_numeric = True
    else:
        current_hue_col = hue_col
        is_numeric = pd.api.types.is_numeric_dtype(df[hue_col])
        if is_numeric and df[hue_col].nunique() < 5:
            is_numeric = False

    if is_numeric:
        palette = "plasma"

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=current_hue_col,
            palette=palette,
            s=150,
            alpha=0.3,
            edgecolor=None,
            legend=False,
            ax=ax
        )

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=current_hue_col,
            palette=palette,
            style='outcome',
            markers=tr_markers,
            s=80,
            alpha=1.0,
            edgecolor='white',
            linewidth=0.6,
            legend=False,
            ax=ax
        )
    else:
        n_hues = plot_df[current_hue_col].nunique()
        palette = sns.color_palette("deep", n_hues) if n_hues <= 10 else sns.color_palette("husl", n_hues)

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=current_hue_col,
            palette=palette,
            s=150,
            alpha=0.2,
            edgecolor=None,
            legend=False,
            ax=ax
        )

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=current_hue_col,
            palette=palette,
            style='outcome',
            markers=tr_markers,
            s=80,
            alpha=0.95,
            edgecolor='white',
            linewidth=0.6,
            ax=ax
        )

        ncol_calc = min(5, n_hues) if n_hues > 0 else 1
        legend = plt.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc='upper center',
            title=hue_col,
            frameon=False,
            ncol=ncol_calc,
            fontsize=18
        )
        if legend is not None:
            plt.setp(legend.get_title(), fontsize=20, fontweight='bold')

    points_all = plot_df[[x_col, y_col]].values
    for _, config in ELLIPSES_CONFIG.items():
        if config['type'] == 'rect':
            x_min, x_max = config['x_range']
            y_min, y_max = config['y_range']
            mask = (
                (plot_df[x_col] >= x_min) & (plot_df[x_col] <= x_max) &
                (plot_df[y_col] >= y_min) & (plot_df[y_col] <= y_max)
            )
        elif config['type'] == 'poly':
            path = mpltPath.Path(config['points'])
            mask = path.contains_points(points_all)
        else:
            continue

        region_df = plot_df[mask]
        if len(region_df) >= 3:
            draw_confidence_ellipse(
                region_df[x_col],
                region_df[y_col],
                ax,
                n_std=config.get('n_std', 2.5),
                edgecolor=config.get('color', '#2E4053')
            )

    for label_cfg in LABELS_CONFIG:
        txt = ax.text(
            x=label_cfg['x'],
            y=label_cfg['y'],
            s=label_cfg['text'],
            fontsize=label_cfg.get('size', 24),
            color=label_cfg.get('color', '#2E4053'),
            fontweight='black',
            horizontalalignment='center',
            verticalalignment='center',
            linespacing=1.2,
            zorder=999
        )
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=5.5, foreground='white', alpha=0.92),
            PathEffects.Normal()
        ])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_color('black')

    plt.xlabel("t-SNE Component 1", fontsize=34, fontweight='black', labelpad=12)
    plt.ylabel("t-SNE Component 2", fontsize=34, fontweight='black', labelpad=12)

    ax.tick_params(
        which='major',
        direction='out',
        length=7,
        width=1.6,
        colors='black',
        top=False,
        right=False,
        left=True,
        bottom=True,
        pad=6,
        labelsize=24
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()


def main():
    """Run the full embedding, projection, and plotting workflow."""
    full_df = load_data_all(data_file, COLUMN_MAPPING)
    if full_df is None:
        return

    for target_cathodes in TARGET_CATHODES_LIST:
        group_name_str = "+".join(target_cathodes)
        print(f"\nProcessing group: {target_cathodes}")

        sub_df = full_df[full_df['cathode'].isin(target_cathodes)].copy()
        n_samples = len(sub_df)

        if n_samples < 3:
            continue

        sentences = []
        for _, row in sub_df.iterrows():
            text = (
                f"Cathode: {row['cathode']}. "
                f"Capacity: {row['capacity']}. "
                f"Format: {row['cell_format']}. "
                f"Trigger: {row['trigger_method']}. "
                f"Electrolyte: {row['electrolyte']}. "
                f"Separator: {row['separator']}. "
                f"Heating Side: {row['heating_side']}. "
                f"Outcome: {row['outcome']}."
            )
            sentences.append(text)

        text_vecs = get_bert_embeddings(sentences)
        if text_vecs is None:
            continue

        num_data = sub_df[['t_trigger_clean', 't_max_clean']].values
        scaler = StandardScaler()
        num_vecs = scaler.fit_transform(num_data)

        final_features = np.hstack([text_vecs, num_vecs * NUMERIC_WEIGHT])

        n_pca = min(50, n_samples - 1)
        pca = PCA(n_components=n_pca)
        pca_res = pca.fit_transform(final_features)

        perp = min(30, max(2, n_samples // 4))
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            init='pca',
            learning_rate='auto'
        )
        coords = tsne.fit_transform(pca_res)

        sub_df['x'] = coords[:, 0]
        sub_df['y'] = coords[:, 1]

        for hue_col in HUE_LIST:
            if hue_col not in sub_df.columns:
                continue

            safe_group_name = group_name_str.replace(" ", "")[:30]
            file_name = f"tsne_{safe_group_name}_{hue_col}_analysis.png"
            save_path = os.path.join(current_dir, file_name)

            plot_tsne(
                sub_df,
                'x',
                'y',
                hue_col,
                save_path,
                group_name_str
            )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
