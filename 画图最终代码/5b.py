import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# ### 用户自定义配置区 ###
# ==========================================

# 1. 文件路径
file_path = r"/Users/ctt/Desktop/电池迁移/数据/5b_filtered.xlsx"

# 2. 指定列名
value_col_name = 'observed_results.max_temperature'

# 3. 自定义想要展示的正极材料列表
target_materials = [
    'NCM111', 'NCM811', 'LFP', 'NCM622', 'Na-Ion', 'NCA', 'NCM-General','NCM523'
]

# 4. Y轴显示范围 (根据原图 a 轴通常为对数或大跨度，这里保持您的 1500)
y_min, y_max = 0, 1500

# ==========================================
# ### 核心绘图逻辑 ###
# ==========================================

# 设置全局学术字体 (Arial)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 读取数据
try:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    if value_col_name not in df.columns:
        raise KeyError(f"Column '{value_col_name}' not found")
except Exception as e:
    print(f"数据读取错误: {e}")
    exit()

df_filtered = df[df['Cathode_Cleaned'].isin(target_materials)].copy()
models = df_filtered.groupby('Cathode_Cleaned')[value_col_name].median().sort_values(ascending=False).index
palette = sns.color_palette("viridis_r", len(models))

# --- 调整比例：改为 Figure 2a 的单面板比例 (约 3:2) ---
fig, ax = plt.subplots(figsize=(10, 4))

# 布局参数
shift_box = -0.2
shift_scatter = 0.2
shift_violin = 0.05
width_box = 0.25
scale_violin = 0.4

for i, model in enumerate(models):
    y_data = df_filtered[df_filtered['Cathode_Cleaned'] == model][value_col_name].dropna().values
    if len(y_data) == 0: continue

    base_color = palette[i]
    fill_color = list(base_color) + [0.4]
    x_center = i

    # A. 箱线图
    ax.boxplot([y_data], positions=[x_center + shift_box], widths=width_box,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=fill_color, edgecolor=base_color, linewidth=1.5),
               medianprops=dict(color='white', linewidth=1.5),
               whiskerprops=dict(color=base_color, linewidth=1.5),
               capprops=dict(color=base_color, linewidth=1.5))

    # B. 散点图 (空心圆)
    jitter = np.random.uniform(-0.08, 0.08, size=len(y_data))
    ax.scatter(np.full_like(y_data, x_center + shift_scatter) + jitter, y_data,
               facecolors='none', edgecolors=base_color, s=25, linewidth=1, zorder=2)

    # C. 密度线
    if len(y_data) > 1:
        kde = stats.gaussian_kde(y_data)
        y_grid = np.linspace(y_data.min(), y_data.max(), 200)
        kde_vals = kde(y_grid)
        kde_vals = kde_vals / kde_vals.max() * scale_violin
        ax.plot(x_center + shift_violin + kde_vals, y_grid, color=base_color, linewidth=2)

# ==============================================================================
# 5. 格式美化 (完全同步 Figure 2a [cite: 143, 193])
# ==============================================================================

ax.set_xticks(range(len(models)))

# X 轴标签：倾斜 30-45 度，字号减小以适应单面板比例
ax.set_xticklabels(
    models,
    rotation=15,
    ha='center',
    rotation_mode='anchor',
    fontsize=16,
    fontweight='bold',
    color='black'
)

# Y 轴标题与刻度
ax.set_ylabel("Max Temperature (°C)", fontsize=18, fontweight='bold', color='black')
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim(y_min, y_max)

# 边框：四周黑边
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

# 网格线：改为实线 (Solid)，浅灰色 [cite: 143]
ax.yaxis.grid(True, linestyle='-', color='#e0e0e0', alpha=0.8, zorder=0)
ax.xaxis.grid(False)

plt.tight_layout()
plt.savefig('Cathode_Fig2a_Style_Replicated.png', dpi=300, bbox_inches='tight')
plt.show()