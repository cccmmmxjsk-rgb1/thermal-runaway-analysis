import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ================= 配置区域 =================
file_path = r"F:\1_15\4i.xlsx"
gas_columns = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']
phenomenon_column = 'observed_results.phenomenon'

# 宏观现象顺序 (指定顺序)
phenomenon_order = [
    'No Fire',
    'Smoking',
    'Jetting',
    'Fire',
    'Explosion'
]

# ---【配色方案：修改为红色系】---
# 【关键修改点】将颜色修改为 "red"，以匹配您上传图片中的红色渐变效果
cmap_style = sns.light_palette("red", as_cmap=True)

try:
    # 1. 读取数据
    df = pd.read_excel(file_path)

    # 2. 数据清洗
    for col in gas_columns:
        # 确保气体列也是数值型，如有非数值转为0或NaN
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. 计算相关性
    if phenomenon_column not in df.columns:
        raise ValueError(f"没找到列名 '{phenomenon_column}'")

    # 对现象列进行 One-Hot 编码 (创建虚拟变量)
    df_dummies = pd.get_dummies(df[phenomenon_column])

    # 合并气体数据和现象的虚拟变量
    df_analysis = pd.concat([df[gas_columns], df_dummies], axis=1)

    # 计算相关系数矩阵
    full_corr = df_analysis.corr()

    # 4. 筛选与切片
    # 只保留指定的现象作为行，气体作为列
    available_phenomena = [p for p in phenomenon_order if p in full_corr.index]
    plot_data = full_corr.loc[available_phenomena, gas_columns]

    # =========================================================
    # 【绘图部分】
    # =========================================================

    # 1. 统一画布大小
    plt.figure(figsize=(6, 5))

    # 2. 绘制热力图
    ax = sns.heatmap(plot_data,
                     annot=True,
                     cmap=cmap_style,  # 使用上面定义的新红色系
                     vmin=-0.2, vmax=0.3,  # 范围保持不变
                     center=None,
                     fmt='.2f',
                     linewidths=0.5,
                     linecolor='lightgrey',
                     cbar=False,
                     square=True,  # 强制正方形格子
                     annot_kws={"size": 10, "color": "black"}
                     )

    # 3. 添加外围黑框 (Spines)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # 4. 添加标题文本
    ax.text(x=0.0, y=1.05, s="i)|phenomenon-gas",
            transform=ax.transAxes,
            fontsize=20, fontweight='bold', color='black',
            ha='left', va='bottom')

    # 5. 轴标签设置
    plt.xlabel("")
    plt.ylabel("")

    # 刻度设置
    plt.xticks(fontsize=11, rotation=45, color='black')
    plt.yticks(fontsize=11, rotation=60, color='black')

    # 6. 紧凑布局
    plt.tight_layout()
    plt.show()

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"发生错误: {e}")