import matplotlib.pyplot as plt
import squarify
import matplotlib
import matplotlib.colors as mcolors

# Input data
raw_data = {
    'NCM-General': 529, 'LFP': 600, 'NCA': 236,
    'NCM523': 254, 'NCM811': 313, 'NCM111': 172,
    'Na-Ion': 108, 'NCM622': 154, 'Li-S': 14, 'OtherS': 77
}

# Split data into NCM-related and non-NCM categories
ncm_data = {k: v for k, v in raw_data.items() if 'NCM' in k}
other_data = {k: v for k, v in raw_data.items() if 'NCM' not in k}

ncm_sorted = dict(sorted(ncm_data.items(), key=lambda item: item[1], reverse=True))
other_sorted = dict(sorted(other_data.items(), key=lambda item: item[1], reverse=True))

# Compute layout proportions
total_count = sum(raw_data.values())
ncm_total = sum(ncm_sorted.values())
height_bottom = (ncm_total / total_count) * 100
height_top = 100 - height_bottom

# Build a unified color map
all_sorted_keys = sorted(raw_data, key=raw_data.get, reverse=True)

if hasattr(matplotlib, 'colormaps'):
    cmap = matplotlib.colormaps['Blues']
else:
    cmap = plt.get_cmap('Blues')

color_map = {}
for i, key in enumerate(all_sorted_keys):
    norm_val = i / len(all_sorted_keys)
    color_map[key] = cmap(0.2 + 0.65 * norm_val)


def calc_plot_data(data_dict, x_start, y_start, width, height):
    """Calculate rectangle positions, colors, labels, and font sizes."""
    values = list(data_dict.values())
    keys = list(data_dict.keys())

    normed_values = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed_values, x_start, y_start, width, height)

    colors = [color_map[k] for k in keys]
    labels = []
    font_sizes = []

    for k, v in zip(keys, values):
        percent = (v / total_count) * 100
        if percent > 15:
            labels.append(f"{k}\n({percent:.1f}%)")
            font_sizes.append(14)
        elif percent > 6:
            labels.append(f"{k}\n({percent:.1f}%)")
            font_sizes.append(11)
        elif percent > 2:
            labels.append(f"{k}")
            font_sizes.append(9)
        else:
            labels.append(f"{k}")
            font_sizes.append(8)

    return rects, colors, labels, font_sizes


# Compute treemap layout for the two groups
rects_bottom, colors_bottom, labels_bottom, fonts_bottom = calc_plot_data(
    ncm_sorted, x_start=0, y_start=0, width=100, height=height_bottom
)

rects_top, colors_top, labels_top, fonts_top = calc_plot_data(
    other_sorted, x_start=0, y_start=height_bottom, width=100, height=height_top
)

all_rects = rects_bottom + rects_top
all_colors = colors_bottom + colors_top
all_labels = labels_bottom + labels_top
all_fonts = fonts_bottom + fonts_top

# Draw figure
plt.figure(figsize=(6, 6), dpi=150)
ax = plt.gca()

x = [r['x'] for r in all_rects]
y = [r['y'] for r in all_rects]
dx = [r['dx'] for r in all_rects]
dy = [r['dy'] for r in all_rects]

ax.bar(
    x, dy, width=dx, bottom=y,
    color=all_colors,
    align='edge',
    edgecolor="#DDDDDD",
    linewidth=1.5
)

for r, lbl, fsize in zip(all_rects, all_labels, all_fonts):
    cx = r['x'] + r['dx'] / 2
    cy = r['y'] + r['dy'] / 2
    ax.text(
        cx, cy, lbl,
        fontsize=fsize,
        fontweight='bold',
        color='black',
        ha='center', va='center'
    )

plt.axis('off')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
