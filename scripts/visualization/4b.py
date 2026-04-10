import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Data Preparation
# ==========================================
# Classifications of thermal runaway outcomes and their respective percentages
labels = ['Jet fire', 'Explosion', 'Smoking', 'Jetting', 'Fire', 'No Fire']
sizes = [0.3, 8.1, 8.8, 15.0, 29.2, 38.5]

# ==========================================
# 2. Color Palette Configuration
# ==========================================
colors = [
    '#d95f5f',  # Jet fire
    '#e07a7e',  # Explosion
    '#be8488',  # Smoking
    '#929292',  # Jetting
    '#7ba7a6',  # Fire
    '#5cb5ac'   # No Fire
]

# ==========================================
# 3. Canvas Initialization
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

# ==========================================
# 4. Pie Chart Rendering
# ==========================================
# startangle=90: Start plotting from the 12 o'clock position
# counterclock=False: Plot in a clockwise direction to match the specified order
wedges, texts, autotexts = ax.pie(sizes,
                                  colors=colors,
                                  startangle=90,
                                  counterclock=False,
                                  autopct='%1.1f%%',
                                  wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
                                  pctdistance=0.65)

# ==========================================
# 5. Internal Label Customization
# ==========================================
for i, autotext in enumerate(autotexts):
    # Format text to display percentage followed by the category label on a new line
    autotext.set_text(f"{sizes[i]}%\n{labels[i]}")

    # Configure font styles
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

    # Special handling: 'Jet fire' proportion (0.3%) is too small to fit internal text.
    # Hide the internal text to avoid overlapping, and prepare for external annotation.
    if labels[i] == 'Jet fire':
        autotext.set_visible(False)

# ==========================================
# 6. External Annotation for Minor Category
# ==========================================
# Calculate the angular position for the 'Jet fire' slice.
# Since it starts at 90 degrees and draws clockwise, the center position is calculated via trig functions.
theta = 90 - (sizes[0] / 2 / 100 * 360)
rad = np.deg2rad(theta)
x = np.cos(rad)
y = np.sin(rad)

# Add external callout line and text
ax.annotate("0.3%\nJet fire",
            xy=(x, y),          # Pointing to the outer edge of the slice
            xytext=(0.1, 1.15), # Text position adjusted for visual clarity
            fontsize=12,
            color='black',
            ha='left',
            arrowprops=dict(arrowstyle="-", color='black', linewidth=1.5))

# ==========================================
# 7. Final Formatting and Export
# ==========================================
# Ensure the pie chart is drawn as a perfect circle
ax.axis('equal')
plt.tight_layout()

# Save the figure in high resolution (300 dpi) for publication purposes
plt.savefig('Thermal_Runaway_Outcomes_PieChart.png', dpi=300, bbox_inches='tight')
plt.show()