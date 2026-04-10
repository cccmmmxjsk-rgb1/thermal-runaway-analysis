import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Data Loading
# ==========================================
file_path = "../data/mass_loss.xlsx"

try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# Column names from the raw dataset (Ensure your CSV headers match these exactly)
col_content = 'mass_loss_content'  # Contains the text description of mass loss
col_count = 'mass_loss_count'      # Contains the frequency/count of occurrences

# ==========================================
# 2. Data Cleansing Function
# ==========================================
def parse_mass_loss(text):
    """
    Parses complex unstructured text entries (e.g., "17-33%", "approx. 75%", "20% (total 1600g)")
    and extracts the standardized mass loss percentage.
    """
    if pd.isna(text):
        return None

    text_str = str(text).lower().strip()
    if not text_str or text_str == 'nan':
        return None

    # Preprocessing: Remove interference characters (%, <, >, approx, etc.)
    clean_text = text_str.replace('%', '')
    clean_text = re.sub(r'[<>\≈\~]', '', clean_text)
    clean_text = clean_text.replace('approx.', '')

    # Strategy 1: Identify ranges (e.g., "10-12" or "10–12") and calculate the average
    range_match = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)', clean_text)
    if range_match:
        try:
            v1 = float(range_match.group(1))
            v2 = float(range_match.group(2))
            if v1 <= 100 and v2 <= 100:
                return (v1 + v2) / 2
        except:
            pass

    # Strategy 2: Extract all numerical values
    numbers = re.findall(r'(\d+\.?\d*)', clean_text)
    numbers = [float(n) for n in numbers if n != '.']

    # Filter out values that are clearly not percentages (e.g., absolute mass in grams)
    valid_percents = [n for n in numbers if 0 <= n <= 100]

    if not valid_percents:
        return None

    # Strategy 3: Logical extraction based on specific keywords
    if 'total' in text_str:
        return max(valid_percents)
    if 'stage' in text_str and len(valid_percents) > 1:
        s = sum(valid_percents)
        if s <= 100:
            return s

    return valid_percents[0]

# Apply parsing function to the dataset
df['parsed_loss'] = df[col_content].apply(parse_mass_loss)

# ==========================================
# 3. Data Expansion
# ==========================================
# Expand the dataset based on the frequency count of each observation
df_clean = df.dropna(subset=['parsed_loss', col_count])
final_values = []

for idx, row in df_clean.iterrows():
    try:
        count = int(row[col_count])
        val = row['parsed_loss']
        if 0 <= val <= 100:
            final_values.extend([val] * count)
    except ValueError:
        continue

final_df = pd.DataFrame({'Mass Loss (%)': final_values})

# ==========================================
# 4. Visualization
# ==========================================
# Initialize canvas with a square aspect ratio
plt.figure(figsize=(8, 8))

# Apply a clean, white background without gridlines for academic formatting
sns.set(style="white")
plt.grid(False)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Define bin edges (0 to 100 with step of 10)
bins_list = list(range(0, 101, 10))

# Plot histogram with Kernel Density Estimate (KDE)
sns.histplot(
    data=final_df,
    x='Mass Loss (%)',
    kde=True,
    stat="percent",
    bins=bins_list,
    color="#4c72b0",
    alpha=0.6,
    shrink=1
)

# Axis labels configuration (Large font size for readability)
plt.xlabel('Mass Loss (%)', fontsize=20, fontweight='bold')
plt.ylabel('Percentage of Cases (%)', fontsize=20, fontweight='bold')

# Tick parameters configuration
plt.tick_params(axis='both', which='major', labelsize=16)

# Lock axis ranges to ensure consistent scaling and avoid visual distortion
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.margins(x=0, y=0)

plt.tight_layout()

# Save the figure in high resolution (300 dpi)
plt.savefig('Mass_Loss_Distribution.png', dpi=300)
plt.show()