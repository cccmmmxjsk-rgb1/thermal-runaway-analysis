import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ================= 配置区域 =================
file_path = r"F:\1_15\ NCM8.xlsx"
gas_columns = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']

# X轴材料
target_materials = ['NCM-General', 'NCM811', 'LFP', 'NCM622', 'Na-Ion', 'NCA', 'NCM111']

# 自定义配色 (保持你的设置)
reference_colors = ['#003366', '#336699', '#85A3C2', '#ECEFF1', '#E09999', '#CC3333', '#990000']
custom_cmap = LinearSegmentedColormap.from_list('reference_style', reference_colors)
# ===========================================

try:
    df = pd.read_excel(file_path)

    # 1. 筛选材料
    df_filtered = df[df['Cathode_Cleaned'].isin(target_materials)]

    if df_filtered.empty:
        print("警告：没有数据")
    else:
        # 2. 数据处理与相关性计算
        df_dummies = pd.get_dummies(df_filtered['Cathode_Cleaned'])

        # 合并 气体数值列 和 材料虚拟变量列
        df_analysis = pd.concat([
            df_filtered[gas_columns].reset_index(drop=True),
            df_dummies.reset_index(drop=True)
        ], axis=1)

        corr_matrix = df_analysis.corr()

        # 3. 筛选与转置
        # 确保只取存在的材料
        available_materials = [m for m in target_materials if m in corr_matrix.index]

        # 结果：行=气体(Y轴)，列=材料(X轴)
        plot_data = corr_matrix.loc[available_materials, gas_columns].T

        # =========================================================
        # 【绘图修改部分：统一风格 (正方形 + 黑框)】
        # =========================================================

        # 1. 统一画布大小
        plt.figure(figsize=(6, 5))

        # 2. 绘制热力图
        ax = sns.heatmap(plot_data,
                         annot=True,
                         cmap=custom_cmap,
                         vmin=-0.3, vmax=0.3,  # 保持你原本设定的 -0.3 到 0.3
                         center=0,
                         fmt='.2f',
                         linewidths=0.5,
                         linecolor='white',  # 内部网格线
                         cbar=False,
                         square=True,  # 【关键】强制正方形格子
                         annot_kws={"size": 10, "color": "black"}  # 数字大小统一为10
                         )

        # 3. 添加外围黑框
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')

        # 4. 添加标题文本 (位置可微调)


        # 5. 轴标签设置
        plt.xlabel("")  # 移除默认标签
        plt.ylabel("")

        # 刻度设置 (统一风格)
        plt.xticks(fontsize=11, rotation=45, color='black')
        plt.yticks(fontsize=11, rotation=60, color='black')  # Y轴改为60度以保持风格一致

        # 6. 紧凑布局
        plt.tight_layout()
        plt.show()

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"发生错误: {e}")