#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_and_plot_ttft_simple(json_file="ttft_wide_dist.json"):
    """
    Simple version: Analyze TTFT vs Input Length using only matplotlib and numpy
    """

    # Read JSON file
    print(f"Reading file: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract data
    ttfts = np.array(data['ttfts'])  # TTFT time (ms)
    input_lens = np.array(data['input_lens'])  # Input length (tokens)

    print(f"Number of data points: {len(ttfts)}")
    print(f"Input length range: {input_lens.min()} - {input_lens.max()} tokens")
    print(f"TTFT range: {ttfts.min():.2f} - {ttfts.max():.2f} ms")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TTFT vs Input Length Analysis', fontsize=16, fontweight='bold')

    # Subplot 1: Scatter plot + trend line
    ax1 = axes[0, 0]
    ax1.scatter(input_lens, ttfts, alpha=0.6, s=20, color='blue')

    # Add trend line
    z = np.polyfit(input_lens, ttfts, 1)
    p = np.poly1d(z)
    ax1.plot(input_lens, p(input_lens), "r--", alpha=0.8, linewidth=2,
             label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')

    ax1.set_xlabel('Input Length (tokens)')
    ax1.set_ylabel('TTFT (ms)')
    ax1.set_title('TTFT vs Input Length - Scatter Plot')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: Average TTFT by input length groups
    ax2 = axes[0, 1]

    # Create groups
    n_bins = 20
    bins = np.linspace(input_lens.min(), input_lens.max(), n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate average TTFT for each bin
    bin_means = []
    bin_stds = []
    for i in range(len(bins)-1):
        mask = (input_lens >= bins[i]) & (input_lens < bins[i+1])
        if np.sum(mask) > 0:
            bin_means.append(ttfts[mask].mean())
            bin_stds.append(ttfts[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    # Remove NaN values
    valid_mask = ~np.isnan(bin_means)
    valid_centers = bin_centers[valid_mask]
    valid_means = bin_means[valid_mask]
    valid_stds = bin_stds[valid_mask]

    ax2.errorbar(valid_centers, valid_means, yerr=valid_stds,
                fmt='o-', capsize=5, capthick=2, alpha=0.8)
    ax2.set_xlabel('Input Length (tokens)')
    ax2.set_ylabel('Average TTFT (ms)')
    ax2.set_title('Average TTFT by Input Length')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Statistics information
    ax3 = axes[1, 0]

    # Calculate correlation and statistics
    correlation = np.corrcoef(input_lens, ttfts)[0, 1]

    # Manual R² calculation
    y_mean = ttfts.mean()
    ss_tot = np.sum((ttfts - y_mean) ** 2)
    ss_res = np.sum((ttfts - p(input_lens)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    stats_text = f"""
Statistical Analysis:

Correlation coefficient: {correlation:.4f}
R² value: {r_squared:.4f}
Regression slope: {z[0]:.6f} ms/token
Regression intercept: {z[1]:.2f} ms

Average TTFT: {ttfts.mean():.2f} ms
TTFT std dev: {ttfts.std():.2f} ms
TTFT median: {np.median(ttfts):.2f} ms

Average input length: {input_lens.mean():.0f} tokens
Input length std dev: {input_lens.std():.0f} tokens
Input length median: {np.median(input_lens):.0f} tokens

P99 TTFT: {np.percentile(ttfts, 99):.2f} ms
P95 TTFT: {np.percentile(ttfts, 95):.2f} ms
P90 TTFT: {np.percentile(ttfts, 90):.2f} ms
    """

    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Statistics Summary')

    # Subplot 4: TTFT distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(ttfts, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(ttfts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ttfts.mean():.2f}ms')
    ax4.axvline(np.median(ttfts), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(ttfts):.2f}ms')
    ax4.set_xlabel('TTFT (ms)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('TTFT Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = json_file.replace('.json', '_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")

    # Show plot (commented out for remote server)
    # plt.show()

    return correlation, r_squared, z[0], z[1]

def print_detailed_analysis(json_file="ttft_wide_dist.json"):
    """
    Print detailed numerical analysis
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    ttfts = np.array(data['ttfts'])
    input_lens = np.array(data['input_lens'])

    print("\n" + "="*50)
    print("Detailed Analysis Report")
    print("="*50)

    # Percentile analysis
    print(f"\nTTFT Percentile Analysis:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(ttfts, p)
        print(f"  P{p}: {value:.2f} ms")

    print(f"\nInput Length Percentile Analysis:")
    for p in percentiles:
        value = np.percentile(input_lens, p)
        print(f"  P{p}: {value:.0f} tokens")

    # TTFT analysis by input length groups
    print(f"\nTTFT Analysis by Input Length Groups:")
    quartiles = np.percentile(input_lens, [25, 50, 75])

    short_mask = input_lens <= quartiles[0]
    medium_mask = (input_lens > quartiles[0]) & (input_lens <= quartiles[2])
    long_mask = input_lens > quartiles[2]

    print(f"  Short input (<={quartiles[0]:.0f} tokens): TTFT = {ttfts[short_mask].mean():.2f} +/- {ttfts[short_mask].std():.2f} ms")
    print(f"  Medium input ({quartiles[0]:.0f}-{quartiles[2]:.0f} tokens): TTFT = {ttfts[medium_mask].mean():.2f} +/- {ttfts[medium_mask].std():.2f} ms")
    print(f"  Long input (>{quartiles[2]:.0f} tokens): TTFT = {ttfts[long_mask].mean():.2f} +/- {ttfts[long_mask].std():.2f} ms")

if __name__ == "__main__":
    # Get JSON file from command line argument or use default
    json_file = sys.argv[1] if len(sys.argv) > 1 else "ttft_wide_dist.json"

    # Execute analysis
    correlation, r_squared, slope, intercept = load_and_plot_ttft_simple(json_file)
    print_detailed_analysis(json_file)

    print(f"\nKey Findings:")
    print(f"- TTFT vs Input Length correlation: {correlation:.4f}")
    print(f"- TTFT increases by {slope:.6f} ms per additional token")
    print(f"- Model explains {r_squared*100:.2f}% of TTFT variance")
