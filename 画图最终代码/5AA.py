import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches

# ==========================================
# ### 用户自定义配置区 ###
# ==========================================

# --- 数据集 1 (左侧: Trigger Temp) ---
file_path_1 = r"F:\1_15\5a.xlsx"
col_name_1 = 'observed_results.T_trigger'

# --- 数据集 2 (右侧: Max Temp) ---
file_path_2 = r"F:\1_15\5b_filtered.xlsx"
col_name_2 = 'observed_results.max_temperature'

# --- 公共配置 ---
target_materials = [
    'NCM-111', 'NCM811', 'LFP', 'Na-Ion', 'NCA', 'NCM-General','NCM-622']

# 【修改点1】Y轴范围调整至 1500
y_min, y_max = 0, 1500

# ==========================================
# ### 数据读取与处理 ###
# ==========================================

def load_data(path, col, targets):
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()
        if col not in df.columns:
            print(f"列名 '{col}' 不存在")
            return None
        return df[df['Cathode_Cleaned'].isin(targets)].copy()
    except Exception as e:
        print(f"错误: {e}")
        return None

df1 = load_data(file_path_1, col_name_1, target_materials)
df2 = load_data(file_path_2, col_name_2, target_materials)

if df1 is None or df2 is None:
    exit()

# 排序：依据 Trigger Temp 的中位数排序
models = df1.groupby('Cathode_Cleaned')[col_name_1].median().sort_values(ascending=False).index

# 颜色设置
palette_trigger = sns.color_palette("mako_r", len(models)) # 冷色
palette_max = sns.color_palette("flare_r", len(models))    # 暖色

# ==========================================
# ### 核心绘图逻辑 ###
# ==========================================

# 设置画布
fig, ax = plt.subplots(figsize=(20, 5))

# 【修改点2】布局参数调整，使图表更宽
scale = 0.6  # 增大整体比例 (原为0.4)
shift_left = -0.25  # 左组中心稍微外移
shift_right = 0.25  # 右组中心稍微外移

# 内部组件偏移与宽度 (根据 scale 动态计算)
offset_box = -0.15 * scale
offset_scatter = 0.15 * scale
offset_kde = 0.05 * scale

width_box = 0.25 * scale  # 增加箱体宽度系数
width_kde = 0.5 * scale   # 增加密度图宽度系数

# 循环绘制
for i, model in enumerate(models):

    # --- 左侧: Trigger Temp ---
    color1 = palette_trigger[i]
    data_1 = df1[df1['Cathode_Cleaned'] == model][col_name_1].dropna().values

    if len(data_1) > 0:
        c_pos = i + shift_left
        # Box
        ax.boxplot([data_1], positions=[c_pos + offset_box], widths=width_box,
                   patch_artist=True, showfliers=False, whis=1.5,
                   boxprops=dict(facecolor=list(color1) + [0.4], edgecolor=color1, linewidth=1.8),
                   medianprops=dict(color='white', linewidth=1.8),
                   whiskerprops=dict(color=color1, linewidth=1.8),
                   capprops=dict(color=color1, linewidth=1.8))
        # Scatter
        jitter = np.random.uniform(-0.05, 0.05, size=len(data_1)) # 抖动稍微增加
        ax.scatter(np.full_like(data_1, c_pos + offset_scatter) + jitter, data_1,
                   facecolors='none', edgecolors=color1, s=20, linewidth=1.2, zorder=2, alpha=0.6)
        # KDE
        if len(data_1) > 1:
            try:
                kde = stats.gaussian_kde(data_1)
                y_grid = np.linspace(data_1.min(), data_1.max(), 100)
                vals = kde(y_grid)
                vals = vals / vals.max() * width_kde
                ax.plot(c_pos + offset_kde + vals, y_grid, color=color1, linewidth=1.8, alpha=0.8)
            except: pass

    # --- 右侧: Max Temp ---
    color2 = palette_max[i]
    data_2 = df2[df2['Cathode_Cleaned'] == model][col_name_2].dropna().values

    if len(data_2) > 0:
        c_pos = i + shift_right
        # Box
        ax.boxplot([data_2], positions=[c_pos + offset_box], widths=width_box,
                   patch_artist=True, showfliers=False, whis=1.5,
                   boxprops=dict(facecolor=list(color2) + [0.4], edgecolor=color2, linewidth=1.8),
                   medianprops=dict(color='white', linewidth=1.8),
                   whiskerprops=dict(color=color2, linewidth=1.8),
                   capprops=dict(color=color2, linewidth=1.8))
        # Scatter
        jitter = np.random.uniform(-0.05, 0.05, size=len(data_2))
        ax.scatter(np.full_like(data_2, c_pos + offset_scatter) + jitter, data_2,
                   facecolors='none', edgecolors=color2, s=20, linewidth=1.2, zorder=2, alpha=0.6)
        # KDE
        if len(data_2) > 1:
            try:
                kde = stats.gaussian_kde(data_2)
                y_grid = np.linspace(data_2.min(), data_2.max(), 100)
                vals = kde(y_grid)
                vals = vals / vals.max() * width_kde
                ax.plot(c_pos + offset_kde + vals, y_grid, color=color2, linewidth=1.8, alpha=0.8)
            except: pass

# ==========================================
# ### 图表美化 ###
# ==========================================

ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=0, ha='center', fontsize=13, fontweight='bold', color='#333333')
ax.set_xlim(-0.6, len(models) - 0.4)

#ax.set_ylabel("Temperature (°C)", fontsize=15, fontweight='bold', color='#333333')
ax.set_ylim(y_min, y_max)
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.2)

# 去除多余边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 添加垂直分隔线
for x in range(len(models) - 1):
    ax.axvline(x + 0.5, color='gray', linestyle=':', alpha=0.3)

# 图例说明颜色含义
legend_patches = [
    mpatches.Patch(color=palette_trigger[len(models) // 2], label='Left: Trigger Temp (Cool Colors)'),
    mpatches.Patch(color=palette_max[len(models) // 2], label='Right: Max Temp (Warm Colors)')
]
#ax.legend(handles=legend_patches, loc='upper right', frameon=False, fontsize=12)

#plt.title("Thermal Runaway Comparison: Trigger vs. Max Temperature", fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()

save_name = 'Wide_DoubleColor_Raincloud_Fixed.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"Finished. Saved as {save_name}")
plt.show()