import os
import warnings

# Suppress warnings and HuggingFace endpoint logs for silent execution
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re
import platform

# ==========================================
# Configuration (Supplementary Material Standard)
# ==========================================
# Relative path to the dataset
FILE_PATH = r"../data/tsne_analysis_data.xlsx"
# Save outputs to the current working directory
SAVE_DIR = r"."
REPORT_FILE = os.path.join(SAVE_DIR, "Deep_Analysis_Report.txt")

TARGET_CATHODES_LIST = [
    ['NCM811', 'NCM-622', 'NCM-General', 'NCM-111', 'NCM523', 'LFP'],
]

HUE_LIST = ['cathode']

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

# ==========================================
# Feature Engineering Configuration
# ==========================================

# Text fields for independent embedding followed by equal-weight averaging
TEXT_EMBED_FIELDS = [
    'trigger_method',
    'electrolyte',
    'separator',
    'atmosphere',
    'pressure',
    'safety_design'
]

# Fields for One-Hot Encoding
ONEHOT_FIELDS = [
    'cathode',       
    'cell_format',   
    'heating_side'   
]

# Numeric fields
NUMERIC_FIELDS = ['capacity_clean']

# Adjustable weights for different feature modalities
TEXT_WEIGHT = 1.0
ONEHOT_WEIGHT = 1.0
NUMERIC_WEIGHT = 1.0


# ==========================================
# Plotting Style Configuration
# ==========================================
def set_english_pub_style():
    font_family = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_family
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['legend.title_fontsize'] = 24
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['svg.fonttype'] = 'none'

set_english_pub_style()


# ==========================================
# Data Cleaning & Preprocessing
# ==========================================
def clean_numeric(val):
    if pd.isna(val) or str(val).lower() in ['unknown', 'nan', 'none']:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    match = re.search(r"(\d+(\.\d+)?)", str(val))
    return float(match.group(1)) if match else 0.0

def load_data_all(file_path, mapping):
    if not os.path.exists(file_path):
        return None

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception:
        return None

    df = df.rename(columns=mapping)

    for internal_name in mapping.values():
        if internal_name not in df.columns:
            df[internal_name] = np.nan

    # Clean numeric columns
    for col in ['capacity', 't_trigger', 't_max']:
        df[f'{col}_clean'] = df[col].apply(clean_numeric)

    # Convert non-clean columns to string
    for col in df.columns:
        if col.endswith('_clean'):
            continue
        df[col] = df[col].fillna('unknown').astype(str)

    return df


# ==========================================
# Text Embedding Computation
# ==========================================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_embedding_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "sentence-transformers/all-mpnet-base-v2"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        return tokenizer, model, device
    except Exception:
        return None, None, None

def get_bert_embeddings(text_list, tokenizer, model, device, batch_size=32):
    if tokenizer is None or model is None:
        return None

    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(**inputs)
            batch_embeddings = mean_pooling(output, inputs['attention_mask']).cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

def build_equal_weight_text_embeddings(df, fields, tokenizer, model, device):
    """
    Generate embeddings for each text field independently, then compute the equal-weight average.
    """
    if len(fields) == 0:
        return None

    field_embeddings = []
    for field in fields:
        texts = [f"{field}: {value}" for value in df[field].astype(str).tolist()]
        emb = get_bert_embeddings(texts, tokenizer, model, device)
        if emb is None:
            return None
        field_embeddings.append(emb)

    stacked = np.stack(field_embeddings, axis=0)
    final_text_embeddings = np.mean(stacked, axis=0)
    return final_text_embeddings


# ==========================================
# Categorical & Numeric Feature Processing
# ==========================================
def build_onehot_features(df, onehot_fields):
    if len(onehot_fields) == 0:
        return None

    onehot_df = pd.get_dummies(
        df[onehot_fields].astype(str),
        columns=onehot_fields,
        prefix=onehot_fields,
        dummy_na=False
    )
    return onehot_df.astype(float).values

def build_numeric_features(df, numeric_fields):
    if len(numeric_fields) == 0:
        return None

    num_data = df[numeric_fields].values.astype(float)
    scaler = StandardScaler()
    return scaler.fit_transform(num_data)

def combine_features(text_features=None, onehot_features=None, numeric_features=None,
                     text_weight=1.0, onehot_weight=1.0, numeric_weight=1.0):
    features = []
    if text_features is not None:
        features.append(text_features * text_weight)
    if onehot_features is not None:
        features.append(onehot_features * onehot_weight)
    if numeric_features is not None:
        features.append(numeric_features * numeric_weight)

    if len(features) == 0:
        return None
    return np.hstack(features)


# ==========================================
# Visualization (t-SNE Plot)
# ==========================================
def plot_tsne(df, x_col, y_col, hue_col, save_path, cathode_group_name):
    plt.figure(figsize=(12, 10))
    sns.set_style("white")
    set_english_pub_style()
    ax = plt.gca()

    CATHODE_ORDER_MAP = {
        'LFP': 0, 'NCM-111': 1, 'NCM523': 2,
        'NCM-General': 3, 'NCM-622': 4, 'NCM811': 5
    }

    plot_df = df.copy()

    if hue_col == 'cathode':
        plot_df['cathode_rank'] = plot_df['cathode'].map(CATHODE_ORDER_MAP)
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
            data=plot_df, x=x_col, y=y_col, hue=current_hue_col,
            palette=palette, s=120, alpha=0.9, edgecolor='white',
            linewidth=0.6, legend=False, ax=ax
        )

        if hue_col == 'cathode':
            norm = plt.Normalize(0, 5)
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Cathode Type (Ordered)", fontsize=22, fontweight='bold')
            cbar.outline.set_visible(False)

            ticks_locs = sorted(CATHODE_ORDER_MAP.values())
            ticks_labels = [k for k, v in sorted(CATHODE_ORDER_MAP.items(), key=lambda item: item[1])]
            cbar.set_ticks(ticks_locs)
            cbar.set_ticklabels(ticks_labels)
            cbar.ax.tick_params(labelsize=18)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontsize(18)
                tick.set_fontweight('bold')
        else:
            norm = plt.Normalize(plot_df[current_hue_col].min(), plot_df[current_hue_col].max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"{hue_col}", fontsize=18)
            cbar.ax.tick_params(labelsize=16)
            cbar.outline.set_visible(False)

    else:
        n_hues = plot_df[current_hue_col].nunique()
        palette = sns.color_palette("deep", n_hues) if n_hues <= 10 else sns.color_palette("husl", n_hues)

        sns.scatterplot(
            data=plot_df, x=x_col, y=y_col, hue=current_hue_col,
            palette=palette, s=120, alpha=0.9, edgecolor='white',
            linewidth=0.6, ax=ax
        )

        ncol_calc = min(5, n_hues) if n_hues > 0 else 1
        plt.legend(
            bbox_to_anchor=(0.5, -0.15), loc='upper center',
            title=hue_col.capitalize(), frameon=False, ncol=ncol_calc,
            fontsize=15, title_fontsize=16
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.xlabel("t-SNE Component 1", fontsize=28, fontweight='bold', labelpad=12)
    plt.ylabel("t-SNE Component 2", fontsize=28, fontweight='bold', labelpad=12)

    ax.tick_params(
        which='major', direction='out', length=6, width=1.5,
        colors='black', top=False, right=False, left=True,
        bottom=True, pad=6, labelsize=18
    )

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as PNG and SVG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()


# ==========================================
# Report Generation
# ==========================================
def write_analysis_to_txt(df, group_name, onehot_dim):
    mode = 'a' if os.path.exists(REPORT_FILE) else 'w'
    detail_cols = [
        'cathode', 'capacity', 'trigger_method', 'cell_format', 'electrolyte',
        'separator', 'atmosphere', 'pressure', 'safety_design', 'heating_side'
    ]

    with open(REPORT_FILE, mode, encoding='utf-8') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"🔬 Deep Analysis Report: {group_name}\n")
        f.write(f"🕒 Time: {pd.Timestamp.now()}\n")
        f.write(f"{'=' * 80}\n")

        f.write("\n>>> Global Summary\n")
        f.write(f"    Samples: {len(df)}\n")
        f.write(f"    Text Embedding Fields (Equal Weight): {', '.join(TEXT_EMBED_FIELDS)}\n")
        f.write(f"    One-Hot Fields: {', '.join(ONEHOT_FIELDS)}\n")
        f.write(f"    One-Hot Dimension: {onehot_dim}\n")
        f.write(f"    Numeric Fields: capacity_clean\n")
        f.write(f"    {'-' * 60}\n")

        if len(df) == 0:
            f.write("    (No samples)\n")
            return

        for col in ['cathode', 'electrolyte', 'separator', 'heating_side', 'trigger_method', 'cell_format']:
            top3 = df[col].value_counts().head(3).to_dict()
            top3_str = ", ".join([f"{k}({v})" for k, v in top3.items()])
            f.write(f"    - Top {col}: {top3_str}\n")

        f.write(f"\n    📋 [Details - {len(df)} items]\n")
        for idx, row in df.iterrows():
            f.write(f"    🔸 [Index {idx}]\n")
            for col_name in detail_cols:
                val = str(row.get(col_name, 'N/A'))
                f.write(f"        • {col_name:<15}: {val}\n")
            f.write("\n")
        f.write(f"    {'=' * 60}\n")


# ==========================================
# Main Execution Flow
# ==========================================
def main():
    if os.path.exists(REPORT_FILE):
        try:
            os.remove(REPORT_FILE)
        except Exception:
            pass

    full_df = load_data_all(FILE_PATH, COLUMN_MAPPING)
    if full_df is None:
        return

    tokenizer, model, device = load_embedding_model()
    if tokenizer is None or model is None:
        return

    for group_idx, target_cathodes in enumerate(TARGET_CATHODES_LIST):
        group_name_str = "+".join(target_cathodes)

        mask = full_df['cathode'].isin(target_cathodes)
        sub_df = full_df[mask].copy()
        n_samples = len(sub_df)

        if n_samples < 3:
            continue

        text_features = build_equal_weight_text_embeddings(
            sub_df, TEXT_EMBED_FIELDS, tokenizer, model, device
        )
        if text_features is None:
            continue

        onehot_features = build_onehot_features(sub_df, ONEHOT_FIELDS)
        if onehot_features is None:
            continue

        numeric_features = build_numeric_features(sub_df, NUMERIC_FIELDS)
        if numeric_features is None:
            continue

        final_features = combine_features(
            text_features=text_features,
            onehot_features=onehot_features,
            numeric_features=numeric_features,
            text_weight=TEXT_WEIGHT,
            onehot_weight=ONEHOT_WEIGHT,
            numeric_weight=NUMERIC_WEIGHT
        )

        n_pca = min(50, n_samples - 1, final_features.shape[1])
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

        write_analysis_to_txt(sub_df, group_name_str, onehot_dim=onehot_features.shape[1])

        for hue_col in HUE_LIST:
            if hue_col not in sub_df.columns:
                continue
            safe_group_name = group_name_str.replace(" ", "")[:30]
            file_name = f"Figure_TSNE_{safe_group_name}_{hue_col}.png"
            plot_tsne(
                sub_df, 'x', 'y', hue_col,
                os.path.join(SAVE_DIR, file_name),
                group_name_str
            )

if __name__ == "__main__":
    main()
