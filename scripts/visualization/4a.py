import matplotlib.pyplot as plt
import numpy as np

# 1. Define the experimental data.
# All chemical species are formatted consistently using LaTeX notation.
# Species such as CO, HF, DMC, and VOC are also enclosed in $...$ for uniform rendering.
labels = [
    r'$CO_2$', r'$CO$', r'$H_2$', r'$CH_4$',
    r'$C_2H_4$', r'$C_2H_6$', r'$HF$', r'$C_3H_6$',
    r'$C_3H_8$', r'$SO_2$', r'$C_2H_2$', r'$NH_3$',
    r'$N_2$', r'$DMC$', r'$EMC$', r'$DEC$',
    r'$C_3H_4O$', r'$VOC$', r'$EC$', r'$C_4H_8$'
]

values = [
    76.8, 74.8, 58.9, 45.8,
    41.7, 26.2, 14.9, 14.4,
    8.7, 6.5, 5.2, 4.1,
    3.4, 3.3, 3.3, 2.8,
    2.6, 2.6, 2.5, 2.5
]

# 2. Compute angular positions for the polar bar chart.
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
width = (2 * np.pi) / N * 0.5

# 3. Create the polar coordinate figure.
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 4. Draw the bars.
inner_radius = 80
ax.set_ylim(0, 180)

bars = ax.bar(angles, values, width=width, bottom=inner_radius,
              color='#F5B041', edgecolor='black', linewidth=0.8)
ax.axis('off')

# 5. Add annotations and formatting details.
tick_len = 5
text_pad = 5

for i, (angle, value, label) in enumerate(zip(angles, values, labels)):
    angle_deg = np.degrees(angle)

    # A. Draw a short radial connector line.
    ax.plot([angle, angle], [inner_radius, inner_radius - tick_len],
            color='black', lw=1)

    # B. Add labels along the inner ring.
    anchor_radius = inner_radius - tick_len - text_pad
    if 0 <= angle < np.pi:
        rotation = 90 - angle_deg
        alignment = 'right'
    else:
        rotation = 90 - angle_deg + 180
        alignment = 'left'

    # Highlight the top five components in bold.
    is_top5 = (i < 5)

    if is_top5:
        # Since all labels are written in LaTeX format ($...$),
        # remove the outer dollar signs and wrap the content with \mathbf{}.
        # Example: r'$CO$' -> r'$\mathbf{CO}$'
        content = label[1:-1]
        display_label = r'$\mathbf{' + content + '}$'
        fw = 'bold'  # Retained for clarity, although bold may depend on LaTeX rendering.
    else:
        display_label = label
        fw = 'normal'

    # Draw the inner-ring label.
    ax.text(angle, anchor_radius, display_label,
            ha=alignment, va='center',
            rotation=rotation, rotation_mode='anchor',
            fontsize=11, fontweight=fw)

    # C. Add percentage values outside the bars.
    if value > 10:
        text_rot = -angle_deg
        if 90 < angle_deg < 100:
            text_rot += 180

        ax.text(angle, inner_radius + value + 2, f"{value}%",
                ha='center', va='bottom',
                rotation=text_rot, rotation_mode='anchor',
                fontsize=10, color='black')

# Draw the reference circle at the inner radius.
x = np.linspace(0, 2 * np.pi, 1000)
ax.plot(x, [inner_radius] * 1000, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()
