import sqlite3
import json
import plotly.graph_objects as go
from collections import defaultdict


def get_dynamic_sankey_data(top_n_raw_words=20):
    """
    Read raw cathode spellings from the database and extract the Top N entries.
    """
    conn = sqlite3.connect("batteries.db")
    cursor = conn.cursor()
    cursor.execute("SELECT specific_material, battery_type FROM experiments")
    rows = cursor.fetchall()
    conn.close()

    # Map raw source spellings to normalized system classes.
    raw_to_clean_map = defaultdict(lambda: defaultdict(int))

    # Count raw spellings so the most frequent terms can be displayed.
    raw_word_counts = defaultdict(int)

    for row in rows:
        mat_raw, b_type = row
        clean_name = f"{str(b_type).upper()} System" if b_type else "OTHER System"

        raw_keyword = "Cathode field not extracted"
        if mat_raw:
            try:
                mat_dict = json.loads(mat_raw)
                cathode = str(mat_dict.get("cathode", "")).strip()
                if cathode:
                    raw_keyword = cathode
            except:
                pass

        raw_to_clean_map[raw_keyword][clean_name] += 1
        raw_word_counts[raw_keyword] += 1

    # Select the Top N raw spellings by frequency.
    sorted_raw_words = sorted(raw_word_counts.items(), key=lambda x: x[1], reverse=True)
    top_raw_words = set([word for word, count in sorted_raw_words[:top_n_raw_words]])

    # Group long-tail spellings outside Top N into one aggregate node.
    final_raw_to_center = defaultdict(int)
    final_center_to_clean = defaultdict(int)

    # Preserve the observed flow from each raw spelling to normalized class.
    top_raw_links = []

    for raw_word, targets in raw_to_clean_map.items():
        for clean_word, count in targets.items():
            if raw_word in top_raw_words:
                display_word = f"'{raw_word}'"
                final_raw_to_center[display_word] += count
                top_raw_links.append((display_word, clean_word, count))
            else:
                display_word = "Other long-tail and sparse names"
                final_raw_to_center[display_word] += count
                top_raw_links.append((display_word, clean_word, count))

            final_center_to_clean[clean_word] += count

    return final_raw_to_center, final_center_to_clean, top_raw_links


# ---------------------------------------------------------
# Number of high-frequency raw spelling nodes to display.
TOP_N = 20
# ---------------------------------------------------------

raw_counts, clean_counts, detailed_links = get_dynamic_sankey_data(TOP_N)

# Build nodes.
raw_nodes = list(raw_counts.keys())
center_node = "ETL normalization layer (app.py)"
clean_nodes = list(clean_counts.keys())

all_nodes = raw_nodes + [center_node] + clean_nodes
node_indices = {name: i for i, name in enumerate(all_nodes)}

sources = []
targets = []
values = []

# (A) Raw spelling -> ETL layer.
for raw_word, count in raw_counts.items():
    if count > 0:
        sources.append(node_indices[raw_word])
        targets.append(node_indices[center_node])
        values.append(count)

# (B) ETL layer -> normalized system class.
for clean_word, count in clean_counts.items():
    if count > 0:
        sources.append(node_indices[center_node])
        targets.append(node_indices[clean_word])
        values.append(count)

# Set colors.
node_colors = []
for name in all_nodes:
    if name in raw_nodes:
        node_colors.append('#EF553B')
    elif name == center_node:
        node_colors.append('#00CC96')
    else:
        node_colors.append('#636EFA')

# Draw the chart.
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=10,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=[f"{n} ({raw_counts[n]})" if n in raw_nodes else (f"{n} ({clean_counts[n]})" if n in clean_nodes else n)
               for n in all_nodes],
        color=node_colors
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color="rgba(180, 180, 180, 0.35)"
    )
)])

fig.update_layout(
    title_text=f"Cathode Material Name Normalization Flow (Top {TOP_N} Raw Source Spellings)",
    font_size=11,
    font_family="Arial",
    width=1200,
    height=800,
    margin=dict(l=50, r=50, t=60, b=20)
)

fig.show()
