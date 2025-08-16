import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def polynomial_fit(x, y, degree):
    """Fit polynomial of given degree to data"""
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)

# Set style for better looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Read the CSV file
df = pd.read_csv('ttft_results_short.csv')

# Filter out data with token count >= 10000
print(f"Original data points: {len(df)}")
df_filtered = df[df['prompt_length'] < 10000]
print(f"Filtered data points (tokens < 10000): {len(df_filtered)}")
print(f"Removed {len(df) - len(df_filtered)} data points with tokens >= 10000")

# Create output directory if it doesn't exist
output_dir = 'ttft_analysis_plots'
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Prepare data using filtered dataset
x = df_filtered['prompt_length'].values
y = df_filtered['ttft_full_prefill'].values

# Create single plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Scatter plot with 4th degree polynomial fit
ax.scatter(x, y, alpha=0.4, s=20, color='lightblue')

# Fit 4th degree polynomial only
x_smooth = np.linspace(x.min(), x.max(), 300)
poly_func = polynomial_fit(x, y, 4)
y_poly = poly_func(x_smooth)
ax.plot(x_smooth, y_poly, color='red', linewidth=2, alpha=0.8)

ax.set_xlabel('Prompt Length (tokens) - Amazon Dataset (Filtered < 10000)', fontsize=12)
ax.set_ylabel('TTFT Full Prefill (seconds)', fontsize=12)
ax.set_title('TTFT Analysis: Relationship between TTFT and Prompt Length (Tokens < 10000)', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Calculate correlation coefficient using filtered data
correlation = np.corrcoef(x, y)[0, 1]

# Show statistics for filtered data
print(f"Filtered data points: {len(df_filtered)}")
print(f"Pearson correlation coefficient: {correlation:.4f}")
print(f"Average TTFT: {df_filtered['ttft_full_prefill'].mean():.4f} seconds")
print(f"Average prompt length: {df_filtered['prompt_length'].mean():.1f} tokens")
print(f"TTFT std deviation: {df_filtered['ttft_full_prefill'].std():.4f} seconds")
print(f"Prompt length range: {df_filtered['prompt_length'].min():.0f} - {df_filtered['prompt_length'].max():.0f} tokens")

# Save the plot
filename = f'ttft_polynomial_analysis_filtered_{timestamp}.png'
filepath = os.path.join(output_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {filepath}")

# Optionally still show the plot (comment out if not needed)
plt.show()
