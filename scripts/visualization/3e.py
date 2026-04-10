import os
import pandas as pd
import plotly.graph_objects as go


def hex_to_rgba(hex_color, opacity):
    """Convert a hex color to an RGBA string with the given opacity."""
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    rgb = tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'


# Load data from a relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "sankey_flow_data.xlsx")

try:
    raw_df = pd.read_excel(file_path)
    print("Excel file loaded successfully.")

    # Define the flow order
    flow_columns = [
        'Category',
        'specific_material.form_factor',
        'specific_material.SOC',
        'thermal_conditions.heating_position'
    ]

    # Extract and clean the required columns
    df = raw_df[flow_columns].copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # Add a hidden tail node to push the last visible labels to the right
    df['Hidden_Tail'] = 'Hidden_Node'
    flow_columns.append('Hidden_Tail')

    # Filter categories using predefined valid values
    valid_filters = {
        'specific_material.SOC': ['0%', '25%', '50%', '75%', '100%', 'other'],
        'Category': ['NCM', 'LFP', 'NCM811', 'LCO', 'NCM523', 'NCA'],
        'thermal_conditions.heating_position': ['Global', 'Side', 'Bottom', 'Wrap', 'Tap', 'Surface', 'Induced'],
        'specific_material.form_factor': ['Cylindrical', 'Pouch', 'Prismatic', 'Coin Cell', 'Mono cell', 'Pellet/Powder']
    }

    print("Filtering data...")
    for col_name, valid_list in valid_filters.items():
        if col_name in df.columns:
            df = df[df[col_name].isin(valid_list)]

    print(f"Filtering completed. {len(df)} valid rows remain.")
    if len(df) == 0:
        print("Warning: no valid data available.")
        raise SystemExit

except Exception as e:
    print(f"Error: {e}")
    raise SystemExit


def generate_sankey_data(df, col_order, palette):
    """Generate node and link data for a Sankey diagram."""
    labels = []
    source = []
    target = []
    value = []
    node_colors = []
    link_colors = []
    unique_nodes_list = []

    # Build nodes
    for i, col in enumerate(col_order):
        items = df[col].unique()
        col_color = palette[i % len(palette)]

        for item in items:
            unique_nodes_list.append((col, item))

            if item == 'Hidden_Node':
                node_colors.append('rgba(0,0,0,0)')
                labels.append("")
            else:
                node_colors.append(col_color)
                labels.append(item)

    node_map = {node: i for i, node in enumerate(unique_nodes_list)}

    # Build links
    for i in range(len(col_order) - 1):
        col_start = col_order[i]
        col_end = col_order[i + 1]

        link_counts = df.groupby([col_start, col_end]).size().reset_index(name='count')

        for _, row in link_counts.iterrows():
            start_node_key = (col_start, row[col_start])
            end_node_key = (col_end, row[col_end])

            if start_node_key in node_map and end_node_key in node_map:
                s_idx = node_map[start_node_key]
                t_idx = node_map[end_node_key]

                source.append(s_idx)
                target.append(t_idx)
                value.append(row['count'])

                source_hex = node_colors[s_idx]

                if row[col_end] == 'Hidden_Node':
                    link_colors.append('rgba(0,0,0,0)')
                else:
                    link_colors.append(hex_to_rgba(source_hex, 0.3))

    return labels, source, target, value, node_colors, link_colors


# Plot settings
color_palette = [
    '#99C5E8',
    '#96D9C6',
    '#B4E3A6',
    '#F2E776',
    '#FFFFFF'
]

labels, sources, targets, values, node_colors, link_colors = generate_sankey_data(
    df, flow_columns, color_palette
)

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=1,
        thickness=20,
        line=dict(color="white", width=0.5),
        label=labels,
        color=node_colors
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors
    )
)])

fig.update_layout(
    font_size=14,
    height=450,
    width=1600,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(t=10, b=10, l=10, r=50)
)

fig.show(
    config={
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'sankey_flow_figure',
            'height': 450,
            'width': 1600,
            'scale': 1
        }
    }
)
