import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

# Input data
raw_data = {
    'Cylindrical': 1412,
    'Prismatic': 641,
    'Pouch': 502,
    'Coin Cell': 43,
    'Pellet/Powder': 27,
    'Others': 38
}

sorted_items = sorted(raw_data.items(), key=lambda item: item[1], reverse=True)
total_val = sum(raw_data.values())

# Build color map
try:
    cmap = mpl.colormaps['Greens']
except AttributeError:
    cmap = plt.cm.get_cmap('Greens')

color_map = {}
num_items = len(sorted_items)

for i, (k, v) in enumerate(sorted_items):
    color_map[k] = cmap(0.35 + 0.55 * (i / (num_items - 1)))

# Compute custom treemap layout
rects = []
width = 100
height = 100

val_cylindrical = raw_data['Cylindrical']
h_bottom = height * (val_cylindrical / total_val)
h_top = height - h_bottom
y_split = h_bottom

rects.append({
    'label': 'Cylindrical', 'value': val_cylindrical,
    'x': 0, 'y': 0, 'dx': width, 'dy': h_bottom
})

top_items = {k: v for k, v in raw_data.items() if k != 'Cylindrical'}
val_top_total = sum(top_items.values())

val_pouch = raw_data['Pouch']
val_prismatic = raw_data['Prismatic']
small_keys = [k for k in sorted(top_items.keys(), key=lambda x: raw_data[x], reverse=True)
              if k not in ['Pouch', 'Prismatic']]
val_small_total = sum(raw_data[k] for k in small_keys)

w_pouch = width * (val_pouch / val_top_total)
w_prismatic = width * (val_prismatic / val_top_total)
w_small = width * (val_small_total / val_top_total)

rects.append({
    'label': 'Pouch', 'value': val_pouch,
    'x': 0, 'y': y_split, 'dx': w_pouch, 'dy': h_top
})

rects.append({
    'label': 'Prismatic', 'value': val_prismatic,
    'x': w_pouch, 'y': y_split, 'dx': w_prismatic, 'dy': h_top
})

current_x = w_pouch + w_prismatic
current_y = y_split

for k in small_keys:
    val = raw_data[k]
    h_item = h_top * (val / val_small_total)
    rects.append({
        'label': k, 'value': val,
        'x': current_x, 'y': current_y, 'dx': w_small, 'dy': h_item
    })
    current_y += h_item

# Draw figure
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
ax.set_aspect('equal')

for r in rects:
    k = r['label']
    percent = (r['value'] / total_val) * 100

    rect_patch = patches.Rectangle(
        (r['x'], r['y']), r['dx'], r['dy'],
        linewidth=1.5,
        edgecolor='#FFFFFF',
        facecolor=color_map[k]
    )
    ax.add_patch(rect_patch)

    text_color = 'white' if percent < 5 else 'black'

    if percent > 40:
        label_text = f"{k}\n({percent:.1f}%)"
        font_size = 20
    elif k == 'Pouch':
        label_text = f"{k}\n({percent:.1f}%)"
        font_size = 19
    elif percent > 20:
        label_text = f"{k}\n({percent:.1f}%)"
        font_size = 16
    elif percent > 2:
        label_text = f"{k}\n{percent:.1f}%"
        font_size = 6
    else:
        label_text = f"{k}\n{percent:.1f}%"
        font_size = 5

    ax.text(
        r['x'] + r['dx'] / 2,
        r['y'] + r['dy'] / 2,
        label_text,
        fontsize=font_size,
        fontweight='bold',
        color=text_color,
        ha='center',
        va='center'
    )

outer_border = patches.Rectangle(
    (0, 0), 100, 100,
    linewidth=4,
    edgecolor='#999999',
    facecolor='none'
)
ax.add_patch(outer_border)

plt.tight_layout()
plt.show()
