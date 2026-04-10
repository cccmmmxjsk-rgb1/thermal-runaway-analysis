import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 1. 读取 Excel 文件
# 请确保文件路径正确
file_path = r"F:\1_15\5q_filtered.xlsx"
df = pd.read_excel(file_path)

# =========================================================
# 步骤 A: 数据清洗
# =========================================================
df.columns.values[0] = 'Material'
df.columns.values[1] = 'Max_Temp'

ni_map = {
    'LFP': 0,
    'NCM-111': 0.33, 'NCM111': 0.33,
    'NCM-523': 0.3, 'NCM523': 0.3,
    'NCM-622': 0.2, 'NCM622': 0.2,
    'NCM-811': 0.1, 'NCM811': 0.1,
    'NCA': 0.1
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
# 使用固定的随机种子可以保证每次生成的抖动一致
np.random.seed(42)
random_spread = (np.random.rand(len(df_plot)) - 0.5) * region_width
df_plot['X_Final'] = df_plot['X_Rank'] + random_spread

# =========================================================
# 步骤 C: 画图
# =========================================================

plt.figure(figsize=(12, 12))
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# ---------------------------------------------------------
# 1. 绘制平滑曲线 (黑色实线) - 【修改点：已改为中位数 median】
# ---------------------------------------------------------
# 这里使用 median() 计算中位数
median_data = df_plot.groupby('X_Rank')['Max_Temp'].median().reset_index().sort_values('X_Rank')
x_median = median_data['X_Rank'].values
y_median = median_data['Max_Temp'].values

# 根据点数决定插值阶数 (点太少时降低阶数)
k_value = 3 if len(x_median) > 3 else (len(x_median) - 1)

if k_value > 0:
    x_smooth = np.linspace(x_median.min(), x_median.max(), 300)
    # 使用中位数数据进行插值
    spl = make_interp_spline(x_median, y_median, k=k_value)
    y_smooth = spl(x_smooth)

    plt.plot(x_smooth, y_smooth,
             color='black',
             linewidth=4,
             linestyle='-',
             alpha=0.6,
             zorder=1)

# 2. 绘制基础散点图 (统一大小)
ax = sns.scatterplot(
    data=df_plot,
    x='X_Final',
    y='Max_Temp',
    hue='Material',
    hue_order=sorted_materials_for_legend,
    palette='plasma_r',
    s=300,                # 统一大小
    alpha=1.0,
    edgecolor=None,
    zorder=2,
    clip_on=False,
    legend=False
)

# 3. 空心圈魔法
path_collection = ax.collections[0]
original_colors = path_collection.get_facecolors()
path_collection.set_edgecolors(original_colors)
path_collection.set_facecolors('none')
path_collection.set_linewidth(2.5)

# =========================================================
# 步骤 D: 坐标轴与美化
# =========================================================

# 1. 设置 X 轴
plt.xticks(
    ticks=range(len(unique_ni_values)),
    labels=unique_ni_values,
    fontweight='bold'
)

# 2. 设置 Y 轴
plt.yticks(
    ticks=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450],
    fontweight='bold'
)
plt.ylim(0, 500)

# 3. 控制刻度外观
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

# 4. 隐藏坐标轴标题
plt.xlabel("")
plt.ylabel("")

# 5. 边框全围起来 + 加粗
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(4)
    spine.set_color('black')

# 6. X 轴范围
plt.xlim(-0.6, len(unique_ni_values) - 0.4)

# =========================================================
# 步骤 E: 单独隐藏 0 位置的刻度线
# =========================================================

x_ticks = ax.xaxis.get_major_ticks()
if len(x_ticks) > 0:
    x_ticks[0].tick1line.set_visible(False)

y_ticks = ax.yaxis.get_major_ticks()
if len(y_ticks) > 0:
    y_ticks[0].tick1line.set_visible(False)

plt.tight_layout()
# 修改文件名以反映逻辑变更
plt.savefig('Bubble_Plot_Fixed_Y_Median.png', dpi=300, bbox_inches='tight')
plt.show()