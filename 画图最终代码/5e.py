import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 1. 读取 Excel 文件
file_path = r"/Users/ctt/Desktop/电池迁移/数据/5q_filtered.xlsx"
df = pd.read_excel(file_path)

# =========================================================
# 步骤 A: 数据清洗
# =========================================================
df.columns.values[0] = 'Material'
df.columns.values[1] = 'Mass_Loss'

# 如果 Mass_Loss 是百分号字符串，转为数值
if df['Mass_Loss'].dtype == 'object':
    df['Mass_Loss'] = df['Mass_Loss'].astype(str).str.rstrip('%').astype(float)

# 材料对应的 Ni 含量映射
ni_map = {
    'LFP': 0,
    'NCM-111': 0.33, 'NCM111': 0.33,
    'NCM-523': 0.5, 'NCM523': 0.5,
    'NCM-622': 0.6, 'NCM622': 0.6,
    'NCM-811': 0.8, 'NCM811': 0.8,
    'NCA': 0.8
}

df['Ni_Content'] = df['Material'].astype(str).str.strip().map(ni_map)
df_plot = df.dropna(subset=['Ni_Content', 'Mass_Loss']).copy()

# =========================================================
# 步骤 B: 计算排名和抖动
# =========================================================
unique_ni_values = sorted(df_plot['Ni_Content'].unique())
rank_map = {val: i for i, val in enumerate(unique_ni_values)}
df_plot['X_Rank'] = df_plot['Ni_Content'].map(rank_map)

sorted_materials_for_legend = (
    df_plot.sort_values('Ni_Content')['Material'].unique().tolist()
)

region_width = 0.75
random_spread = (np.random.rand(len(df_plot)) - 0.5) * region_width
df_plot['X_Final'] = df_plot['X_Rank'] + random_spread

# =========================================================
# 步骤 C: 计算平均值并生成平滑曲线数据
# =========================================================
mean_data = df_plot.groupby('X_Rank')['Mass_Loss'].mean().sort_index()
x_nodes = mean_data.index.values
y_nodes = mean_data.values

if len(x_nodes) > 1:
    k_order = 3 if len(x_nodes) >= 4 else (len(x_nodes) - 1)
    spl = make_interp_spline(x_nodes, y_nodes, k=k_order)
    x_smooth = np.linspace(x_nodes.min(), x_nodes.max(), 300)
    y_smooth = spl(x_smooth)
else:
    x_smooth, y_smooth = x_nodes, y_nodes

# =========================================================
# =========================================================
# 步骤 D: 画图
# =========================================================
fig, ax = plt.subplots(figsize=(8, 8))   # 画布改小一点
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# 平滑曲线
ax.plot(x_smooth, y_smooth, color='black', linewidth=3, zorder=1, alpha=0.9)

# 散点图
sns.scatterplot(
    data=df_plot,
    x='X_Final',
    y='Mass_Loss',
    hue='Material',
    hue_order=sorted_materials_for_legend,
    palette='plasma_r',
    s=300,
    alpha=1.0,
    edgecolor=None,
    zorder=2,
    clip_on=False,
    legend=False,
    ax=ax
)

# 空心圈效果
path_collection = ax.collections[0]
original_colors = path_collection.get_facecolors()
path_collection.set_edgecolors(original_colors)
path_collection.set_facecolors('none')
path_collection.set_linewidth(2.5)

# =========================================================
# 步骤 E: 坐标轴与美化
# =========================================================
ax.set_xticks(range(len(unique_ni_values)))
ax.set_xticklabels(unique_ni_values, fontweight='bold')
ax.tick_params(
    axis='both',
    which='major',
    labelsize=24,
    width=4,
    length=12,
    bottom=True,
    left=True,
    direction='out',
    pad=10
)

ax.set_xlabel("Ni Content (%)", fontsize=36, fontweight='bold', labelpad=20)
ax.set_ylabel("Trigger Tempreture(°C)", fontsize=36, fontweight='bold', labelpad=20)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(4)
    spine.set_color('black')

ax.set_ylim(0,450)
ax.set_xlim(-0.6, len(unique_ni_values) - 0.4)

# 关键：让坐标轴内部绘图区为正方形
ax.set_box_aspect(1)

# 隐藏第一个刻度线
x_ticks = ax.xaxis.get_major_ticks()
#if len(x_ticks) > 0:
    #x_ticks[0].tick1line.set_visible(False)

y_ticks = ax.yaxis.get_major_ticks()
if len(y_ticks) > 0:
    y_ticks[0].tick1line.set_visible(False)

plt.tight_layout()
plt.savefig('Bubble_Plot_Mass_Loss_vs_Ni_Content.png', dpi=300, bbox_inches='tight')
plt.show()
