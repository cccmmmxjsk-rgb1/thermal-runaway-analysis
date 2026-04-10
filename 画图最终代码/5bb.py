import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.colors import LinearSegmentedColormap

# ================= 配置区域 =================

# 1. 文件路径 (请修改为你实际的文件路径)
file_path = r"F:\1_15\5Bb.xlsx"  # 例如: r"C:\Data\data.xlsx"

# 2. 指定 X轴 和 Y轴 对应的 Excel 列名 (根据你的截图设置)
col_x_name = 'SOC_Cleaned'  # X轴变量 (例如截图中的 mass)
col_y_name = 'mass'  # Y轴变量 (例如截图中的 SOC_Cleaned，如果表头不同请修改)

# 3. 指定顺序 (关键步骤)
# 因为 "<10%" 和 "20%~30%" 这种文本，计算机默认排序可能乱序。
# 建议在此处手动列出你希望在 X轴 和 Y轴 显示的顺序。
# 如果列表为空 []，代码将尝试自动排序。

target_x_order = ['0%','0%~50%', '50%~80%','80%~100%', '100%']  # 示例：按逻辑顺序填写X轴类别
target_y_order = ['>60%','40%~60%','20%~30%','10%~20%', '<10%',]  # 示例：按逻辑顺序填写Y轴类别

# 4. 图表尺寸控制
fixed_plot_width = 6
fixed_plot_height = 6

# 5. 边距控制 (单位：英寸)
margin_left = 1.5  # 左侧留给Y轴文字
margin_bottom = 1.5  # 底部留给X轴文字
margin_right = 0.4
margin_top = 0.2

# 6. 自定义配色 (红白蓝)
colors = ["#053061", "#f7f7f7", "#b2182b"]#b2182b
my_red_blue = LinearSegmentedColormap.from_list("custom_rd_bu", colors)

# ================= 代码执行区域 =================

try:
    # 1. 读取数据
    df = pd.read_excel(file_path)

    # 检查列名是否存在
    if col_x_name not in df.columns or col_y_name not in df.columns:
        raise ValueError(f"列名错误：请检查Excel中是否包含 '{col_x_name}' 和 '{col_y_name}'")

    # 2. 数据清洗 (转为字符串并去空格)
    df[col_x_name] = df[col_x_name].astype(str).str.strip()
    df[col_y_name] = df[col_y_name].astype(str).str.strip()

    # 3. 如果没有手动指定顺序，则自动获取唯一值并排序
    if not target_x_order:
        target_x_order = sorted(df[col_x_name].unique())
        print(f"提示：X轴使用自动排序: {target_x_order}")

    if not target_y_order:
        target_y_order = sorted(df[col_y_name].unique())
        print(f"提示：Y轴使用自动排序: {target_y_order}")

    # 4. 计算相关性 (One-Hot Encoding)
    # 将分类变量转换为虚拟变量
    df_x_encoded = pd.get_dummies(df[col_x_name], prefix='', prefix_sep='')
    df_y_encoded = pd.get_dummies(df[col_y_name], prefix='', prefix_sep='')

    # 合并后计算相关系数矩阵
    df_all = pd.concat([df_x_encoded, df_y_encoded], axis=1)
    full_corr = df_all.corr()

    # 5. 筛选数据与转置
    # 我们需要构建一个矩阵：行是Y轴变量，列是X轴变量
    # 确保只提取存在的列/行
    available_x = [x for x in target_x_order if x in full_corr.columns]
    available_y = [y for y in target_y_order if y in full_corr.index]

    if not available_x or not available_y:
        print("警告：无法找到匹配的数据标签，请检查配置区域的顺序列表名称是否与Excel内容一致。")
    else:
        # loc[行标签, 列标签]，这里我们需要 loc[Y轴变量, X轴变量]
        # 这样画出来的图，Y轴就是SOC，X轴就是Mass
        plot_data = full_corr.loc[available_y, available_x]

        # ---------------------------------------------------------
        # 【布局计算】(保持原有的精确控制逻辑)
        # ---------------------------------------------------------
        fig_width = margin_left + fixed_plot_width + margin_right
        fig_height = margin_bottom + fixed_plot_height + margin_top

        fig = plt.figure(figsize=(fig_width, fig_height))

        rect = [
            margin_left / fig_width,
            margin_bottom / fig_height,
            fixed_plot_width / fig_width,
            fixed_plot_height / fig_height
        ]

        ax = fig.add_axes(rect)
        # ---------------------------------------------------------

        # 6. 绘图
        sns.heatmap(plot_data,
                    annot=True,  # 显示数值
                    cmap=my_red_blue,  # 自定义颜色
                    vmin=-0.3, vmax=0.3,  # 【调整】根据相关性强弱调整范围，通常分类相关性数值较小，建议设小一点
                    center=0,
                    fmt='.2f',  # 保留两位小数
                    linewidths=0.5,
                    linecolor='white',
                    cbar=False,  # 关闭色条
                    ax=ax)

        # 边框美化
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')

        # 标签与刻度设置
        plt.xlabel(col_x_name, fontsize=16, labelpad=15)
        plt.ylabel(col_y_name, fontsize=16, labelpad=15)

        plt.xticks(fontsize=14, rotation=30, ha='right')  # X轴旋转
        plt.yticks(fontsize=14, rotation=0)  # Y轴水平

        plt.show()

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"发生错误: {e}")