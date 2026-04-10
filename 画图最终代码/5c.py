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
df.columns.values[2] = 'Mass_Loss'
#df.columns.values[1] = 'Max_Temp'

if df['Mass_Loss'].dtype == 'object':
    df['Mass_Loss'] = df['Mass_Loss'].astype(str).str.rstrip('%').astype('float')

ni_map = {
    'LFP': 0,
    'NCM-523': 0.5, 'NCM523': 0.5,
    'NCM-622': 0.6, 'NCM622': 0.6,
    'NCM-811': 0.8,'NCM811': 0.8,
    'NCA':0.8
}
df['Ni_Content'] = df['Material'].astype(str).str.strip().map(ni_map)
df_plot = df.dropna(subset=['Ni_Content']).copy()

# =========================================================
# 步骤 B: 计算排名和抖动
# =========================================================
unique_ni_values = sorted(df_plot['Ni_Content'].unique())
rank_map = {val: i for i, val in enumerate(unique_ni_values)}
df_plot['X_Rank'] = df_plot['Ni_Content'].map(rank_map)
sorted_materials_for_legend = df_plot.sort_values('Ni_Content')['Material'].unique().tolist()

region_width = 0.75
random_spread = (np.random.rand(len(df_plot)) - 0.5) * region_width
df_plot['X_Final'] = df_plot['X_Rank'] + random_spread

# =========================================================
# (修改点) B2: 计算平均值并生成平滑曲线数据
# =========================================================
# 🔥🔥🔥 修改：将 .median() 改为 .mean()
mean_data = df_plot.groupby('X_Rank')['Max_Temp'].mean().sort_index()
x_nodes = mean_data.index.values
y_nodes = mean_data.values

# 生成平滑曲线 (B-Spline 插值)
if len(x_nodes) > 1:
    k_order = 3 if len(x_nodes) >= 4 else (len(x_nodes) - 1)
    spl = make_interp_spline(x_nodes, y_nodes, k=k_order)
    x_smooth = np.linspace(x_nodes.min(), x_nodes.max(), 300)
    y_smooth = spl(x_smooth)
else:
    x_smooth, y_smooth = x_nodes, y_nodes

# =========================================================
# 步骤 C: 画图
# =========================================================

plt.figure(figsize=(12, 15))
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# --- 绘制平滑曲线 ---
plt.plot(x_smooth, y_smooth, color='black', linewidth=3, zorder=1, alpha=0.9)

# 1. 绘制基础散点图
ax = sns.scatterplot(
    data=df_plot,
    x='X_Final',
    y='Max_Temp',
    hue='Material',
    hue_order=sorted_materials_for_legend,
    palette='plasma_r',
    size='Mass_Loss',
    sizes=(150, 900),
    alpha=1.0,
    edgecolor=None,
    zorder=2,
    clip_on=False,
    legend=False
)

# 2. 空心圈魔法
path_collection = ax.collections[0]
original_colors = path_collection.get_facecolors()
path_collection.set_edgecolors(original_colors)
path_collection.set_facecolors('none')
path_collection.set_linewidth(2.5)

# =========================================================
# 步骤 D: 坐标轴与美化
# =========================================================

# X Ticks
plt.xticks(
    ticks=range(len(unique_ni_values)),
    labels=unique_ni_values,
    fontweight='bold'
)
plt.yticks(fontweight='bold')

plt.tick_params(
    axis='both',
    which='major',
    labelsize=36,
    width=4,
    length=12,
    bottom=True,
    left=True,
    direction='out',
    pad=10
)

# Labels
plt.xlabel("Ni Content (%)", fontsize=40, fontweight='bold', labelpad=20)
plt.ylabel("Trigger Temperature ($^{\circ}$C)", fontsize=40, fontweight='bold', labelpad=20)

# Borders
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(4)
    spine.set_color('black')

plt.ylim(bottom=0)
plt.xlim(-0.6, len(unique_ni_values) - 0.4)

# Hide 0 tick lines
x_ticks = ax.xaxis.get_major_ticks()
if len(x_ticks) > 0:
    x_ticks[0].tick1line.set_visible(False)

y_ticks = ax.yaxis.get_major_ticks()
if len(y_ticks) > 0:
    y_ticks[0].tick1line.set_visible(False)

plt.tight_layout()
plt.savefig('Bubble_Plot_Square_Mean_Curve.png', dpi=300, bbox_inches='tight')
plt.show()