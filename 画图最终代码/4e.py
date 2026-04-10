import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ================= 配置区域 =================
file_path = r"F:\1_15\4e.xlsx"

# X轴材料
target_materials = ['NCM-General', 'NCM811', 'LFP', 'NCM-622', 'Na-Ion', 'NCA', 'NCM111']

# Y轴 Mass 的指定顺序
mass_order = ['<10%', '10%~20%', '20%~30%', '30%~40%', '40%~60%', '>60%']

# 自定义配色
reference_colors = ['#003366', '#336699', '#85A3C2', '#ECEFF1', '#E09999', '#CC3333', '#990000']
custom_cmap = LinearSegmentedColormap.from_list('reference_style', reference_colors)

try:
    df = pd.read_excel(file_path)

    # 1. 筛选 X轴 材料
    df_filtered = df[df['Cathode_Cleaned'].isin(target_materials)].copy()

    # 2. 筛选 Y轴 Mass
    df_filtered = df_filtered[df_filtered['mass'].isin(mass_order)]

    if df_filtered.empty:
        print("警告：没有数据")
    else:
        # 3. 计算相关性 (One-Hot逻辑)
        mass_dummies = pd.get_dummies(df_filtered['mass'])
        mat_dummies = pd.get_dummies(df_filtered['Cathode_Cleaned'])

        # 合并并计算相关矩阵
        full_corr = pd.concat([mass_dummies, mat_dummies], axis=1).corr()

        # 4. 提取绘图数据
        existing_mass = [m for m in mass_order if m in mass_dummies.columns]
        existing_mats = [m for m in target_materials if m in mat_dummies.columns]

        plot_data = full_corr.loc[existing_mass, existing_mats]

        # =========================================================
        # 【绘图修改部分：复刻正方形风格 + 黑框】
        # =========================================================

        # 1. 统一画布大小
        plt.figure(figsize=(6, 5))

        # 2. 绘制热力图
        ax = sns.heatmap(plot_data,
                         annot=True,
                         cmap=custom_cmap,
                         vmin=-0.5, vmax=0.5,
                         center=0,
                         fmt='.2f',
                         linewidths=0.5,
                         linecolor='white',  # 内部网格线颜色
                         cbar=False,
                         square=True,  # 【关键】强制正方形格子
                         annot_kws={"size": 10, "color": "black"}  # 数字字体大小
                         )

        # 3. 添加外围黑框 (Spines)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)  # 边框粗细
            spine.set_color('black')  # 边框颜色

        # 4. 添加标题文本
        # 使用 set_title 并指定位置，或者使用 text (这里为了防止被切掉，调整了位置参数)
        # transform=ax.transAxes 意味着使用相对坐标 (0=左/下, 1=右/上)


        # 5. 轴标签设置
        plt.xlabel("")  # 去掉默认标签
        plt.ylabel("")

        # 刻度字体与旋转 (与参考风格一致)
        plt.xticks(fontsize=11, rotation=45, color='black')
        plt.yticks(fontsize=11, rotation=60, color='black')

        # 6. 紧凑布局 (防止文字被切掉)
        plt.tight_layout()
        plt.show()

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"发生错误: {e}")