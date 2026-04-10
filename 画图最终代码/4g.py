import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

# --- 1. 读取数据 (请使用你的实际路径) ---
file_path = r"F:\1_15\4G.xlsx"
df_raw = pd.read_excel(file_path)

# --- 筛选特定气体 ---
target_gases = ['CO2', 'CO', 'H2', 'CH4', 'C2H4', 'C2H6', 'HF']
df = df_raw[target_gases].copy()

# --- 2. 自动 LaTeX 格式化 ---
def format_gas_name(name):
    new_name = re.sub(r'(\d+)', r'_{\1}', name)
    return f"${new_name}$"

df.rename(columns=lambda x: format_gas_name(x), inplace=True)

# --- 3. 计算相关性 ---
df_inverted = 1 - df
corr = df_inverted.corr()
corr_reversed = corr.iloc[::-1]

mask = np.triu(np.ones_like(corr_reversed, dtype=bool), k=0)
mask = np.flip(mask, axis=1)

# --- 4. 绘图 (调整画布大小以容纳大字体) ---
sns.set_theme(style="white")
plt.figure(figsize=(10, 9))  # 画布稍微加大一点

heatmap = sns.heatmap(corr_reversed,
            mask=mask,
            cmap='RdBu_r',
            vmax=1, vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            cbar=False,
            annot=True,
            fmt=".2f",
            # 【修改点1】：将格子里数字的字体调大到 16 (之前是11)
            annot_kws={"size": 16})

# --- 5. 样式细节 ---
ax = plt.gca()
ax.yaxis.tick_right()
ax.xaxis.tick_bottom()

ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['right'].set_linewidth(2)   # 边框线也稍微加粗一点点
ax.spines['bottom'].set_linewidth(2)

ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True,
               direction='out', color='black', width=2, length=6)
ax.tick_params(axis='y', which='both', left=False, right=True, labelright=True,
               direction='out', color='black', width=2, length=6)

# 【修改点2】：将坐标轴标签(气体名)字体调大到 18 (之前是13)
plt.xticks(rotation=0, fontsize=26, color='black')
plt.yticks(rotation=0, fontsize=26, color='black')

plt.tight_layout()
plt.show()