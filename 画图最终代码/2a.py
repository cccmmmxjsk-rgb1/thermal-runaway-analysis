import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import ScalarFormatter

# ==========================================
# ### 1. 数据准备 ###
# ==========================================

# --- 【配置：使用模拟数据还是真实文件】 ---
USE_MOCK_DATA = False  # 改为 False 以读取您的 Excel 文件
file_path = r"/Users/ctt/Desktop/电池迁移/清洗结果_Python处理/时间统计.xlsx"

if USE_MOCK_DATA:
    np.random.seed(42)
    # 模拟包含跨度较大的数据 (以便展示 Log 轴的效果)
    mock_columns = ['Deepseek-R1', 'Deepseek-V3', 'claude 4', 'Gemini 2.5 Flash',
                    'Gpt-5', 'Gpt-4o', 'Qwen-Plus','Gemini 2.5 Pro','Qwen-Turb']
    data = {}
    for col in mock_columns:
        # 混合生成一些小数值(5-20)和大数值(100-800)
        s1 = np.random.normal(10, 3, 10)
        s2 = np.random.normal(150, 50, 10)
        s3 = np.random.normal(600, 100, 5)  # 偶发高延迟
        data[col] = np.concatenate([s1, s2, s3])
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
else:
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"无法读取文件: {e}")
        exit()

# ==========================================
# ### 2. 核心绘图逻辑 ###
# ==========================================

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']

# 清理列名
df.columns = df.columns.str.strip()
target_models = df.columns.tolist()

# 排序：按中位数排序
medians = df[target_models].median().sort_values(ascending=False)
sorted_models = medians.index.tolist()

palette = sns.color_palette("viridis", len(sorted_models))

# 创建画布
fig, ax = plt.subplots(figsize=(9, 4))  # 稍微加高一点

# 布局参数
shift_box = -0.15
shift_scatter = 0.18
shift_violin = 0.05
width_box = 0.2
scale_violin = 0.35

# 设定Y轴的显示范围 (Log模式下不能为0)
y_view_min = 1  # 略小于1，以便完整显示1的刻度
y_view_max = 1000  # 略大于1000

for i, model in enumerate(sorted_models):
    y_data = df[model].dropna().values
    # 过滤掉 <= 0 的数据 (Log轴无法显示0或负数)
    y_data = y_data[y_data > 0]

    if len(y_data) == 0: continue

    base_color = palette[i]
    fill_color = list(base_color)
    if len(fill_color) == 3:
        fill_color.append(0.4)
    else:
        fill_color = list(base_color[:3]) + [0.4]

    x_center = i

    # A. 箱线图
    ax.boxplot([y_data], positions=[x_center + shift_box], widths=width_box,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=fill_color, edgecolor=base_color, linewidth=1.5),
               medianprops=dict(color='white', linewidth=1.5),
               whiskerprops=dict(color=base_color, linewidth=1.5),
               capprops=dict(color=base_color, linewidth=1.5))

    # B. 散点图
    jitter = np.random.uniform(-0.05, 0.25, size=len(y_data))
    ax.scatter(np.full_like(y_data, x_center + shift_scatter) + jitter, y_data,
               facecolors='none', edgecolors=base_color, s=20, linewidth=1, zorder=2, alpha=0.8)

    # C. 密度线 (针对 Log 轴的特殊处理)
    if len(y_data) > 1:
        try:
            kde = stats.gaussian_kde(y_data)

            # --- 关键修改：使用 logspace 生成网格 ---
            # 这样画出来的曲线在对数轴上才是平滑的
            y_grid = np.logspace(np.log10(max(y_data.min(), 0.1)), np.log10(y_data.max()), 200)

            kde_vals = kde(y_grid)
            kde_vals = kde_vals / kde_vals.max() * scale_violin

            ax.plot(x_center + shift_violin + kde_vals, y_grid, color=base_color, linewidth=2, alpha=0.9)
        except Exception:
            pass

# ==============================================================================
# 3. 格式美化 (Log Axis 核心设置)
# ==============================================================================

ax.set_xticks(range(len(sorted_models)))
ax.set_xticklabels(sorted_models, rotation=15, ha='center', fontsize=11, fontweight='bold')

# --- 核心修改：设置对数坐标轴 ---
ax.set_yscale('log')
ax.set_ylim(y_view_min, y_view_max)

# 手动设置刻度位置
major_ticks = [1, 10, 100, 1000]
ax.set_yticks(major_ticks)

# 手动设置刻度标签 (强制显示为 1, 10, 100, 1000 而不是 10^0, 10^1)
ax.set_yticklabels([str(t) for t in major_ticks], fontsize=11, fontweight='bold')

# 移除次级刻度 (Log轴默认会有很多小刻度，为了整洁可以关掉)
ax.yaxis.set_minor_locator(plt.NullLocator())

# 标签与边框
ax.set_ylabel("Processing Time (s)", fontsize=13, fontweight='bold')

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

# 网格线：针对主刻度画线
ax.yaxis.grid(True, linestyle='-', color='#87CEEB', alpha=0.8, zorder=0)
ax.xaxis.grid(False)

plt.tight_layout()
plt.savefig('Model_Latency_LogScale.png', dpi=300, bbox_inches='tight')
plt.show()