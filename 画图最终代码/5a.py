import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# ### 用户自定义配置区 ###
# ==========================================

# 1. 文件路径
file_path = r"/Users/ctt/Desktop/电池迁移/数据/5a.xlsx"

# 2. 指定 B 列 (数值列) 名称
value_col_name = 'observed_results.T_trigger'

# 3. 指定要参与分析的材料
target_materials = [
    'NCM111', 'NCM811', 'LFP', 'NCM622', 'Na-Ion', 'NCA', 'NCM-General','NCM523'
]

# 4. Y轴显示范围
y_min, y_max = 0, 500

# ==========================================
# ### 核心绘图逻辑 (保持不动) ###
# ==========================================

# 设置全局字体格式，模拟 Figure 2c 的学术风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 1. 读取与清洗数据
try:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    if value_col_name not in df.columns:
        print(f"\n【错误警告】Excel 中找不到列名：'{value_col_name}'")
        raise KeyError(f"Column '{value_col_name}' not found")

except FileNotFoundError:
    print(f"错误：找不到文件，请检查 file_path 是否正确：{file_path}")
    exit()

# 筛选数据
df_filtered = df[df['Cathode_Cleaned'].isin(target_materials)].copy()

# 排序逻辑
models = df_filtered.groupby('Cathode_Cleaned')[value_col_name].median().sort_values(ascending=False).index

# 设置颜色 (Figure 2c 配色由黄绿向深蓝过渡 )
palette = sns.color_palette("viridis_r", len(models))

# 2. 初始化画布 (15, 4 比例符合原图 2c 的扁宽形态 )
fig, ax = plt.subplots(figsize=(15, 4))

# 3. 布局参数
shift_box = -0.2
shift_scatter = 0.2
shift_violin = 0.05
width_box = 0.25
scale_violin = 0.4

# 4. 循环画图
for i, model in enumerate(models):
    y_data = df_filtered[df_filtered['Cathode_Cleaned'] == model][value_col_name].values
    if len(y_data) == 0: continue

    base_color = palette[i]
    fill_color = list(base_color) + [0.4]
    x_center = i

    # A. 箱线图
    ax.boxplot([y_data], positions=[x_center + shift_box], widths=width_box,
               patch_artist=True, showfliers=False, whis=1.5,
               boxprops=dict(facecolor=fill_color, edgecolor=base_color, linewidth=2),
               medianprops=dict(color='white', linewidth=2),
               whiskerprops=dict(color=base_color, linewidth=2),
               capprops=dict(color=base_color, linewidth=2))

    # B. 散点图
    jitter = np.random.uniform(-0.08, 0.08, size=len(y_data))
    ax.scatter(np.full_like(y_data, x_center + shift_scatter) + jitter, y_data,
               facecolors='none', edgecolors=base_color, s=25, linewidth=1, zorder=2)

    # C. 密度线
    if len(y_data) > 1:
        try:
            kde = stats.gaussian_kde(y_data)
            y_grid = np.linspace(y_data.min() - 10, y_data.max() + 10, 200)
            kde_vals = kde(y_grid)
            if kde_vals.max() > 0:
                kde_vals = kde_vals / kde_vals.max() * scale_violin
            ax.plot(x_center + shift_violin + kde_vals, y_grid,
                    color=base_color, linewidth=2.5)
        except Exception:
            pass

# ==============================================================================
# 5. 格式与字体大小优化 (根据 Figure 2c 调整 [cite: 150, 181-188, 194])
# ==============================================================================

# 设置 X 轴刻度
ax.set_xticks(range(len(models)))

# X 轴标签：水平排列 (rotation=0), 粗体, 调整字号至 14
ax.set_xticklabels(models, rotation=0, ha='center', fontsize=18, color='black', fontweight='bold')

# Y 轴标题：粗体, 字号 16 [cite: 150]
ax.set_ylabel("Trigger Temperature (°C)", fontsize=18, fontweight='bold', color='black')

# Y 轴刻度数字：字号 12
ax.tick_params(axis='y', labelsize=16)

# 设置 Y 轴范围
ax.set_ylim(y_min, y_max)

# 边框设置：显示四周黑边，线宽 1.2
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

# 网格线：水平实线，浅灰色
ax.yaxis.grid(True, linestyle='-', color='#e0e0e0', alpha=0.8, zorder=0)
ax.xaxis.grid(False)

plt.tight_layout()
plt.savefig('Cathode_Raincloud_Plot_Fig2c_Style.png', dpi=300, bbox_inches='tight')
plt.show()