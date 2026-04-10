import matplotlib.pyplot as plt
import squarify
import matplotlib.patches as patches
import matplotlib

# Input data
raw_data = {
    'Global': 990,
    'Side': 780,
    'Induced Internal': 107,
    'Bottom': 209,
    'Wrap': 115,
    'Surface': 311,
    'Fire/Jet': 22,
    'Top': 21,
    'Others': 13
}

sorted_items = sorted(raw_data.items(), key=lambda item: item[1], reverse=True)
total_val = sum(raw_data.values())

# Build color map
try:
    cmap = matplotlib.colormaps.get_cmap('Oranges')
except AttributeError:
    cmap = plt.cm.get_cmap('Oranges')

color_map = {}
num_items = len(sorted_items)

for i, (k, v) in enumerate(sorted_items):
    color_map[k] = cmap(0.2 + 0.65 * (i / (num_items - 1)))

# Compute custom layout with Global and Side placed at the bottom
bottom_keys = ['Global', 'Side']
top_keys = [k for k, v in sorted_items if k not in bottom_keys]

sum_bottom = sum(raw_data[k] for k in bottom_keys)
h_bottom = 100 * (sum_bottom / total_val)
h_top = 100 - h_bottom
y_split = h_bottom

rects = []
width = 100

current_x = 0
for k in bottom_keys:
    val = raw_data[k]
    w = width * (val / sum_bottom)
    rects.append({
        'label': k, 'value': val,
        'x': current_x, 'y': 0, 'dx': w, 'dy': h_bottom
    })
    current_x += w

top_values = [raw_data[k] for k in top_keys]
top_rects = squarify.normalize_sizes(top_values, width, h_top)
top_layout = squarify.squarify(top_rects, 0, y_split, width, h_top)

for i, rect in enumerate(top_layout):
    k = top_keys[i]
    rects.append({
        'label': k, 'value': raw_data[k],
        'x': rect['x'], 'y': rect['y'], 'dx': rect['dx'], 'dy': rect['dy']
    })

# Draw figure
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

for r in rects:
    k = r['label']
    percent = (r['value'] / total_val) * 100

    rect_patch = patches.Rectangle(
        (r['x'], r['y']), r['dx'], r['dy'],
        linewidth=1.5,
        edgecolor='#DDDDDD',
        facecolor=color_map[k]
    )
    ax.add_patch(rect_patch)

    label_text = f"{k}"
    font_size = 6

    if percent > 15:
        label_text += f"\n({percent:.1f}%)"
        font_size = 14
    elif percent > 7:
        label_text += f"\n({percent:.1f}%)"
        font_size = 11
    elif percent > 4:
        label_text += f"\n{percent:.1f}%"
        font_size = 9

    ax.text(
        r['x'] + r['dx'] / 2,
        r['y'] + r['dy'] / 2,
        label_text,
        fontsize=font_size,
        fontweight='bold',
        color='black',
        ha='center',
        va='center'
    )

outer_border = patches.Rectangle(
    (0, 0), 100, 100,
    linewidth=3,
    edgecolor='#888888',
    facecolor='none'
)
ax.add_patch(outer_border)

plt.tight_layout()
plt.show()
