import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ["Elsevier", "ACS", "Wiley", "MDPI", "Springer Nature", "RSC", "Other"]
sizes = [62.4, 9.3, 9.3, 8.0, 4.3, 2.6, 3.9]

# Colors
colors = [
    "#cf6d63",  # Elsevier
    "#b9b9bb",  # ACS
    "#9e9e9e",  # Wiley
    "#89aaaa",  # MDPI
    "#74ada7",  # Springer Nature
    "#cd7f79",  # RSC
    "#b38a8b"   # Unknown/Other
]

# 每个标签单独设置字号
font_sizes = {
    "Elsevier": 36,
    "ACS": 32,
    "Wiley": 28,
    "MDPI": 28,
    "Springer Nature": 16,
    "RSC": 16,
    "Other": 16
}

# Figure
fig, ax = plt.subplots(figsize=(14, 11), facecolor="white")
ax.set_facecolor("white")

# Draw pie
wedges, _ = ax.pie(
    sizes,
    colors=colors,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)

# 放在饼图内部的标签
inside_labels = ["Elsevier", "ACS", "Wiley", "MDPI"]

for i, label in enumerate(labels):
    w = wedges[i]
    angle = (w.theta1 + w.theta2) / 2
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))

    if label in inside_labels:
        ax.text(
            0.65 * x,
            0.65 * y,
            f"{sizes[i]:.1f}%\n{label}",
            ha="center",
            va="center",
            fontsize=font_sizes[label],
            fontweight="bold"
        )

# 外部标签位置，也可单独调
external_positions = {
    "Springer Nature": (-1.18, 0.55),
    "RSC": (-1.18, 0.75),
    "Other": (-1.18, 0.98)
}

# 放在外部带引线的标签
for i, label in enumerate(labels):
    if label in external_positions:
        w = wedges[i]
        angle = (w.theta1 + w.theta2) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))

        tx, ty = external_positions[label]

        ax.annotate(
            f"{sizes[i]:.1f}%\n{label}",
            xy=(x, y),
            xytext=(tx, ty),
            ha="right",
            va="center",
            fontsize=font_sizes[label],
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="-",
                color="black",
                lw=1.8
            )
        )

ax.axis("equal")
plt.tight_layout()
plt.show()
