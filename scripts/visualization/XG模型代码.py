import os
import warnings

# Suppress warnings and HuggingFace endpoint logs
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer
import shap

# ============================================================
# Global Plot Settings
# ============================================================
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ============================================================
# Path Configuration for Supplementary Material
# ============================================================
# Relative path to the dataset
DATA_PATH = r'../data/battery_dataset.xlsx'
# Save figures to the current working directory
SAVE_DIR = r'.'

SHAP_IMPORTANCE_FIGSIZE = (9, 7)
SHAP_IMPORTANCE_LAYOUT = dict(left=0.30, right=0.94, bottom=0.16, top=0.84)

# ============================================================
# 1. Data Loading & Cleaning
# ============================================================
df = pd.read_excel(DATA_PATH)

CATHODE_LIST = ['LFP', 'NMC811', 'NMC523', 'NMC622', 'NMC111', 'NCA', 'NMC9XX']
df_all = df[df['cathode'].isin(CATHODE_LIST)].copy()

# Text field cleaning
text_cols = ['mitigation_strategy_raw', 'sensor_location_raw', 'trigger_method_raw']
replacements = {
    'mitigation_strategy_raw': 'no mitigation',
    'sensor_location_raw': 'unknown sensor location',
    'trigger_method_raw': 'unknown trigger method'
}

for col in text_cols:
    df_all[col] = df_all[col].fillna(replacements[col]).astype(str).str.strip()
    df_all[col] = df_all[col].replace({'0': replacements[col], '0.0': replacements[col], '': replacements[col]})

# Feature definition
numerical_features = ['SOC_pct', 'nominal_capacity_Ah', 'cell_mass_g', 'nominal_voltage_V', 'volume_cm3']
categorical_features = ['cathode', 'lithium_salt', 'Separator Coating', 'form_factor']
target = 'T_trigger_C'

required_cols = numerical_features + categorical_features + text_cols + [target]
df_all = df_all[required_cols].dropna(subset=[target]).reset_index(drop=True)

for col in categorical_features:
    df_all[col] = df_all[col].fillna('Unknown').astype(str)

# ============================================================
# 2. Embeddings & PCA
# ============================================================
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# show_progress_bar=False ensures terminal stays clean
mitigation_emb_raw = st_model.encode(df_all['mitigation_strategy_raw'].tolist(), show_progress_bar=False, batch_size=64)
sensor_emb_raw = st_model.encode(df_all['sensor_location_raw'].tolist(), show_progress_bar=False, batch_size=64)
trigger_emb_raw = st_model.encode(df_all['trigger_method_raw'].tolist(), show_progress_bar=False, batch_size=64)

# Data Splitting
TEST_RATIO = 0.2
SEED = 5319
X_base = df_all[numerical_features + categorical_features + text_cols].copy()
y = df_all[target].copy()

train_idx, test_idx = train_test_split(np.arange(len(df_all)), test_size=TEST_RATIO, random_state=SEED)

X_train_base, X_test_base = X_base.iloc[train_idx].copy(), X_base.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

# KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_train_num = pd.DataFrame(knn_imputer.fit_transform(X_train_base[numerical_features]), columns=numerical_features, index=X_train_base.index)
X_test_num = pd.DataFrame(knn_imputer.transform(X_test_base[numerical_features]), columns=numerical_features, index=X_test_base.index)

# PCA
N_DIM = 3
pca_mit = PCA(n_components=N_DIM, random_state=42)
pca_sen = PCA(n_components=N_DIM, random_state=42)
pca_tri = PCA(n_components=N_DIM, random_state=42)

mit_train_pca = pca_mit.fit_transform(mitigation_emb_raw[train_idx])
mit_test_pca = pca_mit.transform(mitigation_emb_raw[test_idx])

sen_train_pca = pca_sen.fit_transform(sensor_emb_raw[train_idx])
sen_test_pca = pca_sen.transform(sensor_emb_raw[test_idx])

tri_train_pca = pca_tri.fit_transform(trigger_emb_raw[train_idx])
tri_test_pca = pca_tri.transform(trigger_emb_raw[test_idx])

mit_col_names = [f'mitigation_emb_{i}' for i in range(N_DIM)]
sen_col_names = [f'sensor_emb_{i}' for i in range(N_DIM)]
tri_col_names = [f'trigger_method_emb_{i}' for i in range(N_DIM)]
embedding_features = mit_col_names + sen_col_names + tri_col_names

X_train_emb = pd.DataFrame(np.hstack([mit_train_pca, sen_train_pca, tri_train_pca]), columns=embedding_features, index=X_train_base.index)
X_test_emb = pd.DataFrame(np.hstack([mit_test_pca, sen_test_pca, tri_test_pca]), columns=embedding_features, index=X_test_base.index)

# Feature Assembly (Full vs Intrinsic)
X_train_cat, X_test_cat = X_train_base[categorical_features].copy(), X_test_base[categorical_features].copy()
X_train_full = pd.concat([X_train_num, X_train_emb, X_train_cat], axis=1)
X_test_full = pd.concat([X_test_num, X_test_emb, X_test_cat], axis=1)

X_train_intr = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_intr = pd.concat([X_test_num, X_test_cat], axis=1)

num_and_embed_features = numerical_features + embedding_features

preprocessor_full = ColumnTransformer([
    ('num', StandardScaler(), num_and_embed_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])
X_train_proc_full = preprocessor_full.fit_transform(X_train_full)
X_test_proc_full = preprocessor_full.transform(X_test_full)
feat_names_full = [n.replace('num__', '').replace('cat__', '') for n in preprocessor_full.get_feature_names_out()]

preprocessor_intr = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])
X_train_proc_intr = preprocessor_intr.fit_transform(X_train_intr)
X_test_proc_intr = preprocessor_intr.transform(X_test_intr)
feat_names_intr = [n.replace('num__', '').replace('cat__', '') for n in preprocessor_intr.get_feature_names_out()]
cat_feature_names_intr = [n for n in feat_names_intr if n not in numerical_features]

DISPLAY_NAMES = {
    'SOC_pct': 'SOC (%)', 'nominal_capacity_Ah': 'Nominal Capacity (Ah)', 'cell_mass_g': 'Cell Mass (g)',
    'nominal_voltage_V': 'Nominal Voltage (V)', 'volume_cm3': 'Volume (cm³)',
    'mitigation_embedding': 'Mitigation Strategy Emb.', 'sensor_location_embedding': 'Sensor Location Emb.',
    'trigger_method_embedding': 'Trigger Method Emb.', 'cathode': 'Cathode Material',
    'lithium_salt': 'Lithium Salt', 'Separator Coating': 'Separator Coating', 'form_factor': 'Form Factor',
}

# ============================================================
# 3. Model Training & Plot 1 (R2 Comparison)
# ============================================================
models = {
    'RF': RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'CatBoost': CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, subsample=0.8, verbose=0, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0),
    'AdaBoost': AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
}

r2_train, r2_test = [], []
for name, model in models.items():
    model.fit(X_train_proc_full, y_train)
    r2_train.append(r2_score(y_train, model.predict(X_train_proc_full)))
    r2_test.append(r2_score(y_test, model.predict(X_test_proc_full)))

fig, ax = plt.subplots(figsize=(7, 6))
x_pos = np.arange(len(models))
bar_width = 0.28
ax.bar(x_pos - bar_width/2, r2_train, bar_width, label='Train', color='#a1d297')
ax.bar(x_pos + bar_width/2, r2_test, bar_width, label='Test', color='#81aab2')

ax.set_ylabel('R²', fontsize=18, fontweight='bold')
ax.set_xlabel('Models', fontsize=18, fontweight='bold')
ax.set_ylim(0.1, 1.05)
ax.set_xticks(x_pos)
ax.set_xticklabels(list(models.keys()), fontsize=16, fontweight='bold', rotation=15)
ax.tick_params(axis='y', labelsize=15)
for spine in ax.spines.values(): spine.set_linewidth(1.8)
ax.tick_params(width=1.8, length=5)
ax.legend(loc='upper right', bbox_to_anchor=(0.85, 1.0), frameon=False, prop={'size': 18, 'weight': 'bold'})
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'Fig1_Models_R2_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 4. Plot 2 (Scatter w/ Marginals)
# ============================================================
xgb_full = models['XGBoost']
y_pred_full = xgb_full.predict(X_test_proc_full)

mae = mean_absolute_error(y_test, y_pred_full)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_full))
r2 = r2_score(y_test, y_pred_full)

fig = plt.figure(figsize=(8, 8))
gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 0:3])
ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

ax_main.scatter(y_test, y_pred_full, color='#4B73E1', alpha=0.6, s=20, edgecolor='#3251B5', linewidth=0.5, zorder=3)
lim_min, lim_max = min(y_test.min(), y_pred_full.min()) - 10, max(y_test.max(), y_pred_full.max()) + 10
ax_main.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.5, zorder=2)
ax_main.set_xlim(lim_min, lim_max)
ax_main.set_ylim(lim_min, lim_max)

ax_histx.hist(y_test, bins=30, color='#4B73E1', edgecolor='white', linewidth=0.5)
ax_histy.hist(y_pred_full, bins=30, color='#4B73E1', edgecolor='white', linewidth=0.5, orientation='horizontal')
ax_histx.axis('off')
ax_histy.axis('off')

for spine in ax_main.spines.values(): spine.set_linewidth(1.5)
ax_main.tick_params(axis='both', which='major', labelsize=13, direction='in', length=6, width=1.5, top=True, right=True)
ax_main.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax_main.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax_main.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0, top=True, right=True)

ax_main.set_xlabel("True $T_{trigger}$ (°C)", fontsize=16, fontweight='bold')
ax_main.set_ylabel("Predicted $T_{trigger}$ (°C)", fontsize=16, fontweight='bold')
ax_main.text(0.05, 0.95, f"MAE = {mae:.3f} °C\nRMSE = {rmse:.3f} °C\n$R^2$ = {r2:.3f}", transform=ax_main.transAxes, fontsize=14, verticalalignment='top')

plt.savefig(os.path.join(SAVE_DIR, 'Fig2_Scatter_Marginals.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 5. SHAP Calcs & Plot 3 (Beeswarm - Intrinsic)
# ============================================================
xgb_intr = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
xgb_intr.fit(X_train_proc_intr, y_train)

explainer_intr = shap.TreeExplainer(xgb_intr)
shap_values_intr = explainer_intr.shap_values(X_test_proc_intr)
shap_abs_intr = np.abs(shap_values_intr)

num_indices = [feat_names_intr.index(col) for col in numerical_features]
shap_values_num_only = shap_values_intr[:, num_indices]
X_test_num_display = pd.DataFrame(X_test_proc_intr[:, num_indices], columns=[DISPLAY_NAMES.get(col, col) for col in numerical_features])

plt.figure(figsize=(9, 5))
shap.summary_plot(shap_values_num_only, X_test_num_display, max_display=len(numerical_features), show=False)
plt.title("SHAP Beeswarm — Numerical Features (Intrinsic Model)", fontsize=13, fontweight='bold')
plt.xlabel("SHAP value (°C)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'Fig3_SHAP_Beeswarm_Intrinsic.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 6. Plot 4 (SHAP Feature Importance - Full)
# ============================================================
explainer_full = shap.TreeExplainer(xgb_full)
shap_values_full = explainer_full.shap_values(X_test_proc_full)
shap_abs_full = np.abs(shap_values_full)

grouped_shap_full = {}
for col in numerical_features: grouped_shap_full[col] = shap_abs_full[:, feat_names_full.index(col)].mean()
for group_name, col_names in [('mitigation_embedding', mit_col_names), ('sensor_location_embedding', sen_col_names), ('trigger_method_embedding', tri_col_names)]:
    grouped_shap_full[group_name] = shap_abs_full[:, [feat_names_full.index(c) for c in col_names]].sum(axis=1).mean()
for cat_col in categorical_features:
    ohe_cols = [c for c in feat_names_full if c.startswith(cat_col + '_')]
    grouped_shap_full[cat_col] = shap_abs_full[:, [feat_names_full.index(c) for c in ohe_cols]].sum(axis=1).mean()

grouped_full_series = pd.Series(grouped_shap_full).sort_values(ascending=True)

def get_shap_color_full(feat_name):
    if feat_name in numerical_features: return '#e74c3c'
    if 'mitigation' in feat_name: return '#f39c12'
    if 'sensor' in feat_name: return '#27ae60'
    if 'trigger_method' in feat_name: return '#16a085'
    if feat_name == 'cathode': return '#8e44ad'
    return '#3498db'

fig_full, ax_f = plt.subplots(figsize=SHAP_IMPORTANCE_FIGSIZE)
fig_full.subplots_adjust(**SHAP_IMPORTANCE_LAYOUT)
colors_f = [get_shap_color_full(f) for f in grouped_full_series.index]
grouped_full_series.plot(kind='barh', ax=ax_f, color=colors_f, edgecolor='white')

for i, val in enumerate(grouped_full_series.values):
    ax_f.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=10)

ax_f.set_xlabel('Mean |SHAP value| (°C)', fontsize=13)
ax_f.set_title('SHAP Feature Importance — Full Model', fontsize=13, fontweight='bold')
ax_f.grid(True, alpha=0.3, axis='x')

legend_full = [
    Patch(facecolor='#e74c3c', label='Numerical'), Patch(facecolor='#8e44ad', label='Cathode (sum)'),
    Patch(facecolor='#f39c12', label='Mitigation Emb'), Patch(facecolor='#27ae60', label='Sensor Emb'),
    Patch(facecolor='#16a085', label='Trigger Method Emb'), Patch(facecolor='#3498db', label='Other Categorical')
]
ax_f.legend(handles=legend_full, loc='lower right', fontsize=9, frameon=False)
plt.savefig(os.path.join(SAVE_DIR, 'Fig4_SHAP_Grouped_Full.png'), dpi=300)
plt.close()

# ============================================================
# 7. Plot 5 (SHAP Feature Importance - Intrinsic)
# ============================================================
grouped_shap_intr = {}
for col in numerical_features: grouped_shap_intr[col] = shap_abs_intr[:, feat_names_intr.index(col)].mean()
for cat_col in categorical_features:
    ohe_cols = [c for c in cat_feature_names_intr if c.startswith(cat_col + '_')]
    grouped_shap_intr[cat_col] = shap_abs_intr[:, [feat_names_intr.index(c) for c in ohe_cols]].sum(axis=1).mean()

grouped_intr_series = pd.Series(grouped_shap_intr).sort_values(ascending=True)

def get_color_intr(feat_name):
    if feat_name in numerical_features: return '#e74c3c'
    if feat_name == 'cathode': return '#8e44ad'
    return '#3498db'

fig_intr, ax_i = plt.subplots(figsize=SHAP_IMPORTANCE_FIGSIZE)
fig_intr.subplots_adjust(**SHAP_IMPORTANCE_LAYOUT)
display_labels = [DISPLAY_NAMES.get(f, f) for f in grouped_intr_series.index]
colors_i = [get_color_intr(f) for f in grouped_intr_series.index]
ax_i.barh(display_labels, grouped_intr_series.values, color=colors_i, edgecolor='white', linewidth=1.2)

for i, val in enumerate(grouped_intr_series.values):
    ax_i.text(val + 0.15, i, f'{val:.2f}', va='center', fontsize=11, fontweight='bold')

ax_i.set_xlabel('Mean |SHAP value| (°C)', fontsize=13, fontweight='bold')
ax_i.set_title('SHAP Feature Importance — Intrinsic Model', fontsize=14, fontweight='bold')
ax_i.grid(True, alpha=0.3, axis='x')
ax_i.tick_params(axis='both', labelsize=12)

legend_intr = [
    Patch(facecolor='#e74c3c', label='Numerical Feature'),
    Patch(facecolor='#8e44ad', label='Cathode Material'),
    Patch(facecolor='#3498db', label='Other Categorical'),
]
ax_i.legend(handles=legend_intr, loc='lower right', fontsize=10, frameon=False)
for spine in ax_i.spines.values(): spine.set_linewidth(1.5)

plt.savefig(os.path.join(SAVE_DIR, 'Fig5_SHAP_Grouped_Intrinsic.png'), dpi=300)
plt.close()