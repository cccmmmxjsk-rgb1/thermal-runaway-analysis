import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
labels = ['Jet fire', 'Explosion', 'Smoking', 'Jetting', 'Fire', 'No Fire']
sizes = [0.3, 8.1, 8.8, 15.0, 29.2, 38.5]

# 2. 定义颜色 (根据图片吸取近似色值)
colors = [
    '#d95f5f',  # Jet fire (虽然切片很小，给它一个红色系的底色)
    '#e07a7e',  # Explosion (浅红/肉粉色)
    '#be8488',  # Smoking (暗粉/褐玫瑰色)
    '#929292',  # Jetting (灰色)
    '#7ba7a6',  # Fire (灰青色)
    '#5cb5ac'  # No Fire (蓝绿色/青色)
]

# 3. 创建画布
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

# 4. 绘制饼图
# startangle=90: 从12点钟方向开始
# counterclock=False: 顺时针方向绘制 (符合图中 Jet fire -> Explosion -> ... 的顺序)
wedges, texts, autotexts = ax.pie(sizes,
                                  colors=colors,
                                  startangle=90,
                                  counterclock=False,
                                  autopct='%1.1f%%',  # 初始格式，后面会覆盖
                                  wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},  # 白色边框
                                  pctdistance=0.65)  # 文字距离圆心的距离

# 5. 自定义内部文字样式
for i, autotext in enumerate(autotexts):
    # 设置显示的文本格式：百分比 + 换行 + 类别名
    autotext.set_text(f"{sizes[i]}%\n{labels[i]}")

    # 设置字体样式
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

    # 特殊处理：'Jet fire' 占比太小，隐藏内部文字，改为外部引线标注
    if labels[i] == 'Jet fire':
        autotext.set_visible(False)

# 6. 添加 'Jet fire' 的外部引线标注
# 计算第一块切片(Jet fire)的角度位置
# 因为是从90度开始顺时针画，且它只有0.3%，它位于约89.5度的地方(接近12点钟偏右)
theta = 90 - (sizes[0] / 2 / 100 * 360)  # 角度
rad = np.deg2rad(theta)
x = np.cos(rad)
y = np.sin(rad)

# 添加标注
ax.annotate("0.3%\nJet fire",
            xy=(x, y),  # 箭头指向的位置 (圆周边缘)
            xytext=(0.1, 1.15),  # 文字放置的位置 (根据图片目测调整)
            fontsize=12,
            color='black',
            ha='left',  # 水平对齐
            arrowprops=dict(arrowstyle="-", color='black', linewidth=1.5))

# 7. 保持圆形并显示
ax.axis('equal')
plt.tight_layout()
plt.show()