import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 配置区域 =================
file_path = r"F:\1_15\ NCM8.xlsx"

# 1. 气体列（X轴）
gas_columns = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']

# 2. 分类列（Y轴，正极材料）
category_column = 'Cathode_Cleaned'

# 3. 【新增】在这里指定你只想要的材料名字
#    请确保名字和 Excel 里的完全一样（区分大小写）
target_materials = [
    'NCM-General', 'NCM811', 'LFP', 'NCM-622', 'Na-Ion', 'NCA', 'NCM-111']
# ===========================================

try:
    # 读取数据
    df = pd.read_excel(file_path)

    # 【新增】关键步骤：筛选数据
    # 只保留 target_materials 列表中出现过的材料
    df_filtered = df[df[category_column].isin(target_materials)]

    # 检查筛选结果是否为空（防止名字写错导致画不出图）
    if df_filtered.empty:
        print("警告：筛选后没有数据！请检查 target_materials 里的名字是否和 Excel 里的一模一样。")
    else:
        # 计算出现频率
        heatmap_data = df_filtered.groupby(category_column)[gas_columns].mean()

        # 画图
        plt.figure(figsize=(8, 8))  # 可以根据材料多少调整高度，比如 (10, 4)
        sns.heatmap(heatmap_data,annot=True, cmap='Reds', fmt='.2f',linewidths=0.5,cbar_kws={'label': 'Frequency'})

        #plt.title('Correlation between Cathode Material and Output Gas (Frequency)', fontsize=16)
        plt.ylabel('Cathode Material', fontsize=12)
        plt.xlabel('Gas Type', fontsize=12)
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"发生错误: {e}")