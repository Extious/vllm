import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def load_latency_data(file_path="latency_results_diagram_format.csv"):
    """Load latency data from CSV file"""
    if not os.path.exists(file_path):
        print(f"Error: Data file {file_path} not found")
        return None
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records from {file_path}")
    
    # Filter data to only include prompt length below 5000
    df_filtered = df[df['prompt_length'] <= 5000].copy()
    print(f"Filtered to {len(df_filtered)} records with prompt length <= 5000 tokens")
    
    return df_filtered

def plot_latency_vs_total_items_with_prompt_info(df):
    """Plot TTFT and TTLF vs total items with average prompt length info"""
    
    # Calculate average prompt length for each total_items value
    avg_prompt_by_items = df.groupby('total_items')['prompt_length'].mean().round(1)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Fit curves for TTFT
    z_ttft = np.polyfit(df['total_items'], df['ttft'], 2)
    p_ttft = np.poly1d(z_ttft)
    x_smooth = np.linspace(df['total_items'].min(), df['total_items'].max(), 100)
    plt.plot(x_smooth, p_ttft(x_smooth), 'b-', linewidth=3, label='TTFT')
    
    # Fit curves for TTLF
    z_ttlf = np.polyfit(df['total_items'], df['ttlf'], 2)
    p_ttlf = np.poly1d(z_ttlf)
    plt.plot(x_smooth, p_ttlf(x_smooth), 'r-', linewidth=3, label='TTLF')
    
    # Add 100ms annotation line
    plt.axhline(y=0.1, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(plt.xlim()[0], 0.1, '100ms ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='gray')
    
    # Customize plot
    plt.xlabel('Total Items (avg prompt length)', fontsize=12)
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.title('TTFT and TTLF vs Total Items', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add average prompt length information to x-axis labels
    unique_items = sorted(df['total_items'].unique())
    x_ticks = []
    x_labels = []
    
    # Show every 3rd tick to reduce density
    step = max(1, len(unique_items) // 8)  # Show approximately 8 ticks
    
    for i, item_count in enumerate(unique_items):
        if i % step == 0 or i == len(unique_items) - 1:  # Show first, every step-th, and last
            avg_prompt = avg_prompt_by_items.get(item_count, 0)
            x_ticks.append(item_count)
            x_labels.append(f'{item_count}\n({avg_prompt})')
    
    plt.xticks(x_ticks, x_labels, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('latency_vs_total_items_with_prompt.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return plt.gcf()

def main():
    """Main function to run the visualization analysis"""
    print("Loading latency data...")
    df = load_latency_data()
    
    if df is None:
        print("Failed to load data. Please check if the CSV file exists.")
        return
    
    print("Creating latency vs total items plot with prompt length info...")
    plot_latency_vs_total_items_with_prompt_info(df)
    
    print("\nAnalysis complete! Generated plot:")
    print("- latency_vs_total_items_with_prompt.png")

if __name__ == "__main__":
    main() 