import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
file_path = r"F:\1_15\质量损失lv.csv"
try:
    df = pd.read_csv(file_path)
except:
    df = pd.read_csv(file_path, encoding='gbk')

col_content = 'observed_results.mass_loss_percent_内容'
col_count = 'observed_results.mass_loss_percent_次数'


# 2. 清洗函数 (处理复杂文本如 "17-33%", "approx. 75%", "20% (total 1600g)")
def parse_mass_loss_v2(text):
    if pd.isna(text):
        return None

    text_str = str(text).lower().strip()
    if not text_str or text_str == 'nan':
        return None

    # 预处理：去掉干扰字符，保留数字、小数点、连字符
    # 先去掉 % 号，防止正则匹配区间时受阻
    clean_text = text_str.replace('%', '')
    clean_text = re.sub(r'[<>\≈\~]', '', clean_text)
    clean_text = clean_text.replace('approx.', '')

    # 1. 识别区间 (Range) 例如 "10-12" 或 "10–12"，取平均值
    range_match = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)', clean_text)
    if range_match:
        try:
            v1 = float(range_match.group(1))
            v2 = float(range_match.group(2))
            if v1 <= 100 and v2 <= 100:
                return (v1 + v2) / 2
        except:
            pass

    # 2. 提取所有数字
    numbers = re.findall(r'(\d+\.?\d*)', clean_text)
    numbers = [float(n) for n in numbers if n != '.']

    # 过滤掉显然不是百分比的大数 (比如质量 g)
    valid_percents = [n for n in numbers if 0 <= n <= 100]

    if not valid_percents:
        return None

    # 3. 逻辑判断
    # 如果有 "total"，取合法的最大值；如果有 "stage" 且总和<100，取和；否则取第一个
    if 'total' in text_str:
        return max(valid_percents)
    if 'stage' in text_str and len(valid_percents) > 1:
        s = sum(valid_percents)
        if s <= 100: return s

    return valid_percents[0]


# 应用解析
df['parsed_loss'] = df[col_content].apply(parse_mass_loss_v2)

# 3. 展开数据 (根据'次数'列重复数据)
df_clean = df.dropna(subset=['parsed_loss', col_count])
final_values = []
for idx, row in df_clean.iterrows():
    try:
        count = int(row[col_count])
        val = row['parsed_loss']
        if 0 <= val <= 100:
            final_values.extend([val] * count)
    except ValueError:
        continue

final_df = pd.DataFrame({'Mass Loss (%)': final_values})

# 4. 绘图
# 4. 绘图
# 修改1：把 figsize 改为宽长方形 (10, 5)，让柱子物理上变扁
plt.figure(figsize=(8, 8))

# 修改前
# sns.set(style="whitegrid") # 这种风格自带灰色网格线

# 修改后
sns.set(style="white")      # 这种风格是纯白背景
plt.grid(False)             # 双重保险，强制关闭网格
#sns.despine()               # 去掉图表顶部和右侧的边框线，看起来更清爽
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

bins_list = list(range(0, 101, 10))

sns.histplot(
    data=final_df,
    x='Mass Loss (%)',
    kde=True,
    stat="percent",
    bins=bins_list,
    color="#4c72b0",
    alpha=0.6,
    shrink=1  # 修改2：增加柱子间隙
)

#plt.title('Distribution of Battery Mass Loss (Percentage)', fontsize=14, fontweight='bold')
#plt.xlabel('Mass Loss (%)', fontsize=12)
#plt.ylabel('Percentage of Cases (%)', fontsize=12)
#plt.axvline(50, color='#c44e52', linestyle='--', linewidth=2, label='Level 4 Threshold (50%)')

# ==========================================
# ★★★ 在这里添加 plt.ylim !!!
# ==========================================
# 这里的 50 意思是 Y 轴最高显示到 50%。
# 如果您的柱子最高只有 30%，那么这就留出了 20% 的空白，柱子看起来就扁了。
# 1. 强制 X 轴起点锁定为 0 (终点为 100)
# 1. 放大横纵轴标题 (去掉注释并调大 fontsize)
# 建议大小：16-20
plt.xlabel('Mass Loss (%)', fontsize=20, fontweight='bold')
plt.ylabel('Percentage of Cases (%)', fontsize=20, fontweight='bold')

# 2. 放大横纵轴的刻度数字 (0, 10, 20... 这些数字)
# 建议大小：14-16
plt.tick_params(axis='both', which='major', labelsize=16)

# ==========================================

# 锁定坐标轴范围
plt.xlim(0, 100)
plt.margins(x=0, y=0)
plt.ylim(0, 30)

# 如果有图例，也建议放大图例文字
plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig('mass_loss_distribution_v3_flat_large_font.png', dpi=300)
plt.show()