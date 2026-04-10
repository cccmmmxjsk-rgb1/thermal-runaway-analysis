import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ================= 配置区域 =================
file_path = r"F:\1_15\4d.xlsx"
col_soc = 'observed_results.phenomenon'
col_material = 'Cathode_Cleaned'

# X轴材料
target_materials =  ['NCM-General', 'NCM811', 'LFP', 'NCM-622', 'Na-Ion', 'NCA', 'NCM111']
# Y轴现象
target_phenomena = ['No Fire', 'Smoking', 'Jetting','Fire',  'Explosion']

# ---【自定义配色】(保留原有的红白蓝配色) ---
colors = ["#b2182b", "#f7f7f7", "#053061"]
my_red_blue = LinearSegmentedColormap.from_list("custom_rd_bu", colors)

try:
    # 1. 读取数据
    df = pd.read_excel(file_path)

    # 2. 数据清洗
    df[col_soc] = df[col_soc].astype(str).str.strip()
    df[col_material] = df[col_material].astype(str).str.strip()

    replace_dict = {
        '0': '0%', '1': '100%', '1.0': '100%', '0.0': '0%'
    }
    df[col_soc] = df[col_soc].replace(replace_dict)

    # 3. 计算相关性
    df_soc_encoded = pd.get_dummies(df[col_soc], prefix='', prefix_sep='')
    df_mat_encoded = pd.get_dummies(df[col_material], prefix='', prefix_sep='')

    df_all = pd.concat([df_mat_encoded, df_soc_encoded], axis=1)
    full_corr = df_all.corr()

    # 4. 筛选与转置
    available_materials = [m for m in target_materials if m in full_corr.index]
    available_phenomena = [p for p in target_phenomena if p in full_corr.columns]

    if not available_materials or not available_phenomena:
        print("警告：缺少目标材料或现象数据")
    else:
        # 行=现象(Y轴)，列=材料(X轴)
        plot_data = full_corr.loc[available_materials, available_phenomena].T

        # =========================================================
        # 【关键修改部分：完全复刻代码 B 的绘图风格】
        # =========================================================

        # 1. 使用与参考代码一致的画板大小
        plt.figure(figsize=(6, 5))

        # 2. 绘图
        ax = sns.heatmap(plot_data,
                         annot=True,  # 显示数值
                         fmt='.2f',  # 保留两位小数
                         cmap=my_red_blue,  # 保持原本的红白蓝配色逻辑
                         vmin=-0.15, vmax=0.15,
                         center=0,
                         cbar=False,  # 关闭色条
                         linewidths=0.5,  # 保留一点网格线更清晰，如需完全一致可改为0
                         linecolor='white',
                         square=True,  # 【重点】强制单元格为正方形，与代码B一致
                         annot_kws={"size": 10, "color": "black"}  # 【重点】数字字体大小设为10
                         )

        # 3. 去掉默认标签（参考代码B的做法）
        plt.title("")
        plt.xlabel("")
        plt.ylabel("")

        # 4. 刻度字体与旋转调整（完全复刻代码B）
        # 字体大小改为 11，X轴旋转 45度，Y轴旋转 60度
        plt.xticks(fontsize=11, rotation=45, color='black')
        plt.yticks(fontsize=11, rotation=60, color='black')

        # 5. 边框加粗逻辑（复刻代码B）
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')

        # 6. 紧凑布局
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"发生错误: {e}")