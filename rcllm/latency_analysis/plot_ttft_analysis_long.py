import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def polynomial_fit(x, y, degree):
    """Fit polynomial of given degree to data with error handling"""
    try:
        # Remove any NaN or infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < degree + 1:
            print(f"Warning: Not enough data points for {degree}-degree polynomial. Using linear fit.")
            degree = 1
        
        coeffs = np.polyfit(x_clean, y_clean, degree)
        return np.poly1d(coeffs)
    except np.linalg.LinAlgError:
        print(f"Warning: Polynomial fit failed for degree {degree}. Trying lower degree...")
        if degree > 1:
            return polynomial_fit(x, y, degree - 1)
        else:
            print("Error: Even linear fit failed. Returning constant function.")
            return np.poly1d([y.mean()])

def clean_data(df):
    """Clean data by removing outliers and extreme values"""
    # Remove rows with NaN or infinite values
    df_clean = df.dropna(subset=['prompt_length', 'ttft_full_prefill'])
    
    # Remove extreme outliers (beyond 3 standard deviations)
    prompt_mean = df_clean['prompt_length'].mean()
    prompt_std = df_clean['prompt_length'].std()
    ttft_mean = df_clean['ttft_full_prefill'].mean()
    ttft_std = df_clean['ttft_full_prefill'].std()
    
    # Filter outliers
    mask = (
        (df_clean['prompt_length'] >= prompt_mean - 3 * prompt_std) &
        (df_clean['prompt_length'] <= prompt_mean + 3 * prompt_std) &
        (df_clean['ttft_full_prefill'] >= ttft_mean - 3 * ttft_std) &
        (df_clean['ttft_full_prefill'] <= ttft_mean + 3 * ttft_std)
    )
    
    df_filtered = df_clean[mask]
    
    print(f"Original data: {len(df)} records")
    print(f"After cleaning: {len(df_filtered)} records")
    print(f"Removed {len(df) - len(df_filtered)} outlier records")
    
    return df_filtered

# Set style for better looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Read both CSV files
print("Reading ttft_results_long.csv...")
df_long = pd.read_csv('ttft_results_long.csv')

print("Reading ttft_results_retry.csv...")
df_retry = pd.read_csv('ttft_results_retry.csv')

# Combine the datasets
print("Combining datasets...")
df_combined = pd.concat([df_long, df_retry], ignore_index=True)

print(f"Long dataset: {len(df_long)} records")
print(f"Retry dataset: {len(df_retry)} records")
print(f"Combined dataset: {len(df_combined)} records")

# Clean the data
print("Cleaning data...")
df_clean = clean_data(df_combined)

# Create output directory if it doesn't exist
output_dir = 'ttft_analysis_plots'
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Prepare data
x = df_clean['prompt_length'].values
y = df_clean['ttft_full_prefill'].values

# Create single plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Scatter plot with polynomial fit
ax.scatter(x, y, alpha=0.4, s=20, color='lightblue', label='Data points')

# Fit polynomial with error handling
print("Fitting polynomial...")
poly_func = polynomial_fit(x, y, 4)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_poly = poly_func(x_smooth)
ax.plot(x_smooth, y_poly, color='red', linewidth=2, alpha=0.8, label=f'Polynomial fit (degree {len(poly_func.coef)-1})')

ax.set_xlabel('Prompt Length (tokens) - Combined Dataset', fontsize=12)
ax.set_ylabel('TTFT Full Prefill (seconds)', fontsize=12)
ax.set_title('TTFT Analysis: Relationship between TTFT and Prompt Length (Long + Retry)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

# Calculate correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]

# Show statistics
print(f"Cleaned data points: {len(df_clean)}")
print(f"Pearson correlation coefficient: {correlation:.4f}")
print(f"Average TTFT: {df_clean['ttft_full_prefill'].mean():.4f} seconds")
print(f"Average prompt length: {df_clean['prompt_length'].mean():.1f} tokens")
print(f"TTFT std deviation: {df_clean['ttft_full_prefill'].std():.4f} seconds")
print(f"Prompt length range: {df_clean['prompt_length'].min():.0f} - {df_clean['prompt_length'].max():.0f} tokens")

# Save the plot
filename = f'ttft_polynomial_analysis_combined_{timestamp}.png'
filepath = os.path.join(output_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {filepath}")

# Optionally still show the plot (comment out if not needed)
plt.show()