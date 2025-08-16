#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def create_final_regression_plot(csv_path):
    """Create final regression plot with custom x-axis labels and removed CPU Cache curve"""
    # Load data
    df = pd.read_csv(csv_path).dropna()
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Define colors and labels (removed CPU Cache TTFT)
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    labels = ['Full Prefill TTFT', 'GPU Cache TTFT', 'Total CPU Overhead']
    y_columns = ['full_prefill_ttft', 'gpu_cache_ttft', 'total_cpu_overhead']
    
    # Use total_item_number as X axis
    X = df['total_item_number'].values.reshape(-1, 1)
    x_smooth = np.linspace(df['total_item_number'].min(), df['total_item_number'].max(), 200).reshape(-1, 1)
    
    print("=== Final Cubic Polynomial Regression Analysis ===\n")
    
    # Create cubic polynomial regression curves for each metric
    for i, (y_col, color, label) in enumerate(zip(y_columns, colors, labels)):
        y = df[y_col].values
        
        # Create cubic polynomial model (degree=3 for all)
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ])
        
        # Fit model
        model.fit(X, y)
        y_pred_smooth = model.predict(x_smooth)
        r2 = model.score(X, y)
        
        # Print results
        print(f"{label}: R² = {r2:.4f}")
        
        # Plot only regression curve (no data points)
        plt.plot(x_smooth.flatten(), y_pred_smooth, color=color, linewidth=3, 
                label=label)
    
    # Calculate average prompt length for each total_item_number
    item_avg_prompt = df.groupby('total_item_number')['prompt_length'].mean().round(0).astype(int)
    
    # Get data range
    min_items = df['total_item_number'].min()
    max_items = df['total_item_number'].max()
    
    # Generate multiples of 5 within the data range
    tick_items = []
    start_tick = ((min_items - 1) // 5 + 1) * 5  # Round up to next multiple of 5
    end_tick = (max_items // 5) * 5  # Round down to previous multiple of 5
    
    for item_count in range(start_tick, end_tick + 1, 5):
        if item_count <= max_items:
            tick_items.append(item_count)
    
    # Create regression model for prompt length prediction
    prompt_X = df['total_item_number'].values.reshape(-1, 1)
    prompt_y = df['prompt_length'].values
    prompt_model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])
    prompt_model.fit(prompt_X, prompt_y)
    
    # Create custom x-tick labels with format: "item_count (avg_prompt_length)"
    x_tick_labels = []
    for item_count in tick_items:
        if item_count in item_avg_prompt:
            # Use real data
            avg_prompt = item_avg_prompt[item_count]
        else:
            # Use predicted data
            predicted_prompt = prompt_model.predict([[item_count]])[0]
            avg_prompt = int(round(predicted_prompt))
        
        x_tick_labels.append(f"{item_count} ({avg_prompt})")
    
    # Set custom x-axis ticks and labels for multiples of 5
    plt.xticks(tick_items, x_tick_labels, rotation=45, ha='right')
    
    # Customize plot
    plt.xlabel('Total Items (Average Prompt Length)', fontsize=14, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=14, fontweight='bold')
    plt.title('Inference Performance vs Total Items on A100', fontsize=16, fontweight='bold')
    
    # Legend positioning (no dataset info box to conflict with)
    plt.legend(fontsize=12, loc='upper left', framealpha=0.95, borderpad=0.5)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits
    plt.xlim(df['total_item_number'].min() - 1, df['total_item_number'].max() + 1)
    
    # No dataset information box as requested
    
    plt.tight_layout()
    plt.savefig('inference_performance_vs_total_items_A100.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal cubic polynomial regression analysis completed for {len(df)} users")
    print(f"Removed CPU Cache TTFT curve, custom x-axis labels added")
    print(f"X-axis shows: item_count (average_prompt_length)")

def main():
    csv_path = 'multi_user_performance_results_A100.csv'
    try:
        create_final_regression_plot(csv_path)
        print("File saved: 'inference_performance_vs_total_items_A100.png'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
