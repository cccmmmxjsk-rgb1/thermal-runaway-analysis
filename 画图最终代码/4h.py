import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ================= 配置区域 =================
file_path = r"F:\1_15\4h.xlsx"
gas_columns = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']
mass_column = 'mass'

# 质量区间顺序 (保持不变)
mass_order = [
    '<10%',
    '10%~20%',
    '20%~30%',
    '30%~40%',
    '40%~60%',
    '>60%'
]

# ---【配色方案：保留你指定的深青蓝】---
# 既然老弟你要求颜色不变，这里严格使用你代码里的深青色渐变
cmap_style = sns.light_palette("#002b41", as_cmap=True)

try:
    # 1. 读取数据
    df = pd.read_excel(file_path)

    # 2. 数据清洗
    for col in gas_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. 计算相关性
    if mass_column not in df.columns:
        raise ValueError(f"没找到列名 '{mass_column}'")

    df_dummies = pd.get_dummies(df[mass_column])
    df_analysis = pd.concat([df[gas_columns], df_dummies], axis=1)
    full_corr = df_analysis.corr()

    # 4. 筛选与切片
    available_mass = [m for m in mass_order if m in full_corr.index]
    plot_data = full_corr.loc[available_mass, gas_columns]

    # =========================================================
    # 【绘图修改部分：统一为正方形黑框风格】
    # =========================================================

    # 1. 统一画布大小
    plt.figure(figsize=(6, 5))

    # 2. 绘制热力图 (使用你的 cmap_style)
    ax = sns.heatmap(plot_data,
                annot=True,
                cmap=cmap_style,     # 【关键】这里保留了你的深青色
                vmin=-0.2, vmax=0.3, # 范围保持你设定的
                center=None,         # 单色渐变通常不需要 center=0
                fmt='.2f',
                linewidths=0.5,
                linecolor='lightgrey', # 配合深色系，内部网格线用浅灰或白均可
                cbar=False,
                square=True,         # 【关键】强制正方形格子
                annot_kws={"size": 10, "color": "black"} # 统一数字大小
               )

    # 3. 添加外围黑框 (Spines)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # 4. 添加标题文本
    # 根据文件名推测是 h，位置调整到左上角
    ax.text(x=0.0, y=1.05, s="h)|massloss-gas",
            transform=ax.transAxes,
            fontsize=20, fontweight='bold', color='black',
            ha='left', va='bottom')

    # 5. 轴标签设置
    plt.xlabel("")
    plt.ylabel("")

    # 刻度设置 (统一风格：字体11，旋转角度一致)
    plt.xticks(fontsize=11, rotation=45, color='black')
    plt.yticks(fontsize=11, rotation=60, color='black')

    # 6. 紧凑布局
    plt.tight_layout()
    plt.show()

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"发生错误: {e}")