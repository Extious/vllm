import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_data(csv_filepath, output_dir):
    """
    Reads data from a CSV file and generates various plots.

    Args:
        csv_filepath (str): Path to the input CSV file.
        output_dir (str): Directory to save the generated plots.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at {csv_filepath}")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_filepath)

    # Calculate ratios
    # Avoid division by zero if 'ttft_with_full_prefill' or 'input_ids_length' can be zero
    df['ttft_ratio'] = df.apply(lambda row: row['ttft_with_cache'] / row['ttft_with_full_prefill'] if row['ttft_with_full_prefill'] != 0 else 0, axis=1)
    df['length_ratio'] = df.apply(lambda row: row['position_length'] / row['input_ids_length'] if row['input_ids_length'] != 0 else 0, axis=1)

    # Plot 1: TTFT vs Input IDs Length
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='input_ids_length', y='ttft_with_cache', label='TTFT with Cache', alpha=0.7)
    sns.scatterplot(data=df, x='input_ids_length', y='ttft_with_full_prefill', label='TTFT with Full Prefill', alpha=0.7)
    plt.title('TTFT vs Input IDs Length')
    plt.xlabel('Input IDs Length')
    plt.ylabel('TTFT (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ttft_vs_input_ids_length.png'))
    plt.close()
    print(f"Saved plot: ttft_vs_input_ids_length.png to {output_dir}")

    # Plot 2: TTFT vs Position Length
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='position_length', y='ttft_with_cache', label='TTFT with Cache', alpha=0.7)
    sns.scatterplot(data=df, x='position_length', y='ttft_with_full_prefill', label='TTFT with Full Prefill', alpha=0.7)
    plt.title('TTFT vs Position Length')
    plt.xlabel('Position Length (Number of Important Tokens)')
    plt.ylabel('TTFT (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ttft_vs_position_length.png'))
    plt.close()
    print(f"Saved plot: ttft_vs_position_length.png to {output_dir}")

    # Plot 3: TTFT with Cache vs TTFT with Full Prefill
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='ttft_with_cache', y='ttft_with_full_prefill', alpha=0.7)
    # Add a y=x line for reference
    min_val = min(df['ttft_with_cache'].min(), df['ttft_with_full_prefill'].min())
    max_val = max(df['ttft_with_cache'].max(), df['ttft_with_full_prefill'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x (Cache = Full Prefill)')
    plt.title('TTFT with Cache vs TTFT with Full Prefill')
    plt.xlabel('TTFT with Cache (seconds)')
    plt.ylabel('TTFT with Full Prefill (seconds)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is equal
    plt.savefig(os.path.join(output_dir, 'ttft_cache_vs_full_prefill.png'))
    plt.close()
    print(f"Saved plot: ttft_cache_vs_full_prefill.png to {output_dir}")

    # Plot 4: Histogram of Input IDs Length
    plt.figure(figsize=(10, 6))
    sns.histplot(df['input_ids_length'], kde=True, bins=20)
    plt.title('Distribution of Input IDs Length')
    plt.xlabel('Input IDs Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'input_ids_length_distribution.png'))
    plt.close()
    print(f"Saved plot: input_ids_length_distribution.png to {output_dir}")

    # Plot 5: Histogram of Position Length
    plt.figure(figsize=(10, 6))
    sns.histplot(df['position_length'], kde=True, bins=20)
    plt.title('Distribution of Position Length')
    plt.xlabel('Position Length (Number of Important Tokens)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'position_length_distribution.png'))
    plt.close()
    print(f"Saved plot: position_length_distribution.png to {output_dir}")

    # Plot 6: Average TTFT Comparison
    avg_ttft_cache = df['ttft_with_cache'].mean()
    avg_ttft_full_prefill = df['ttft_with_full_prefill'].mean()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['TTFT in our system', 'TTFT in vllm'], y=[avg_ttft_cache, avg_ttft_full_prefill])
    plt.title('Average TTFT Comparison')
    plt.ylabel('Average TTFT (seconds)')
    for i, v in enumerate([avg_ttft_cache, avg_ttft_full_prefill]):
        plt.text(i, v + 0.005, f"{v:.4f}", color='black', ha="center")
    plt.savefig(os.path.join(output_dir, 'average_ttft_comparison.png'))
    plt.close()
    print(f"Saved plot: average_ttft_comparison.png to {output_dir}")

    # Plot 7: TTFT Ratio vs Length Ratio
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='length_ratio', y='ttft_ratio', alpha=0.7)
    plt.title('TTFT Ratio vs. Important Token Ratio')
    plt.xlabel('Position Length / Input IDs Length (Important Token Ratio)')
    plt.ylabel('TTFT with Cache / TTFT with Full Prefill (TTFT Ratio)')
    plt.grid(True)
    # Add a horizontal line at y=1 for reference (where cache TTFT equals full prefill TTFT)
    plt.axhline(1, color='r', linestyle='--', label='TTFT Ratio = 1 (Cache = Full Prefill)')
    
    # Add y=x line
    # Determine the limits for the y=x line based on data range
    min_val_ratio = min(df['length_ratio'].min(), df['ttft_ratio'].min())
    max_val_ratio = max(df['length_ratio'].max(), df['ttft_ratio'].max())
    # Extend slightly for better visualization if necessary, or cap at reasonable bounds (e.g., 0 to 1 or slightly more)
    plot_min = max(0, min_val_ratio - 0.1) # Ensure plot starts at or above 0
    plot_max = min(max(1.1, max_val_ratio + 0.1), 1.5) # Cap max for clarity, ensure it covers at least up to 1.1

    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'g--', lw=2, label='y=x (TTFT Ratio = Length Ratio)')
    
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ttft_ratio_vs_length_ratio.png'))
    plt.close()
    print(f"Saved plot: ttft_ratio_vs_length_ratio.png to {output_dir}")

    print(f"\nAll plots saved to {output_dir}")

if __name__ == "__main__":
    # Assuming the script is in vllm/ and the CSV is in vllm/results/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "results", "ttft_results.csv")
    plot_output_dir = os.path.join(current_dir, "results", "plots")
    
    plot_data(csv_file, plot_output_dir)

