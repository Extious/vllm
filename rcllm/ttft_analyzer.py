import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class TTFTAnalyzer:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.5):
        """Initialize the TTFT analyzer with model and tokenizer"""
        print("Initializing model and tokenizer...")
        self.llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm.set_tokenizer(self.tokenizer)
        print("Model and tokenizer initialized successfully!")

    def load_user_data(self, user_id):
        """Load user historical purchase data and candidates from reviewer data files"""
        # Construct path to the reviewer data file
        data_file_path = os.path.join(os.path.dirname(__file__), "../../../amazon/dataset/reviewer_data_processed", f"{user_id}.json")

        # Check if file exists
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Reviewer data file not found: {data_file_path}")

        # Read the complete data from the reviewer file
        with open(data_file_path, 'r') as f:
            all_data = json.load(f)

        # Ensure we have enough data
        if len(all_data) < 4:
            raise ValueError(f"Insufficient data for {user_id}: need at least 4 items, got {len(all_data)}")

        # Split data: first 3 items as history, rest as candidates
        history_data = all_data[:3]
        candidate_data = all_data[3:]

        return history_data, candidate_data

    def create_recommendation_prompt(self, history_data, candidate_data):
        """Create the recommendation prompt"""
        # Construct the basic prompt prefix
        prefix = f"""You are an intelligent assistant that can rank items based on the user's preference. The history items and candidate items are listed below. The prefix of history items should be [history] and the prefix of candidate items should be [i]. i is the identifier of the candidate item. Please rank the candidate items based on the user's history. You should strictly obey the following rules: 
        1. All the candidate items should be included and listed using identifiers, in descending order of the user's preference. The most preferred recommendation item should be listed first.
        2. The results format should be [] > [], where each [] is an identifier, e.g., [2] > [1] > [0].
        3. Only respond with the ranking results, do not say any word or explain.
        4. Output in the following JSON format: \n{{\"rank\": \"[] > [] .. > []\"}}."""
        
        # Format the historical data as part of the prompt
        purchase_history = "\n".join([f"- {item}" for item in history_data])
        history_prompt = f"{prefix}\n{purchase_history}\n\n"
        
        # Add candidate items
        for i, item in enumerate(candidate_data):
            history_prompt += f"[{i}]: {json.dumps(item)}\n"
        
        # Create the query prompt
        query_prompt = f"""\n{len(history_data)} history items and {len(candidate_data)} candidate items are listed above. Be careful that the number of the ranking items is {len(candidate_data)}. Please give me the JSON format data of the ranking results, do not output anything other than the JSON format data. The ranking results should be in the following format: \n{{\"rank\": \"[] > [] .. > []\"}}."""
        
        return history_prompt + query_prompt

    def measure_ttft_multiple_runs(self, prompt, num_runs=5):
        """Measure TTFT multiple times for statistical analysis"""
        ttft_values = []
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)
        
        print(f"Running {num_runs} TTFT measurements...")
        print(f"Prompt length: {prompt_length} tokens")
        
        sampling_params = SamplingParams(temperature=0, max_tokens=256)
        
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...", end=" ")
            
            # Generate with full prefill (no cache)
            output = self.llm.generate([prompt], sampling_params)
            
            # Calculate TTFT
            ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
            ttft_values.append(ttft)
            
            print(f"TTFT: {ttft:.4f}s")
            
            # Small delay between runs to avoid potential caching effects
            time.sleep(0.1)
        
        return ttft_values, prompt_length

    def analyze_ttft_statistics(self, ttft_values):
        """Analyze TTFT statistics"""
        ttft_array = np.array(ttft_values)
        
        stats = {
            'mean': np.mean(ttft_array),
            'median': np.median(ttft_array),
            'std': np.std(ttft_array),
            'min': np.min(ttft_array),
            'max': np.max(ttft_array),
            'q25': np.percentile(ttft_array, 25),
            'q75': np.percentile(ttft_array, 75),
            'cv': np.std(ttft_array) / np.mean(ttft_array) * 100  # Coefficient of variation
        }
        
        return stats

    def visualize_ttft_results(self, ttft_values, user_id, save_plot=True):
        """Create visualization of TTFT results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: TTFT values over runs
        ax1.plot(range(1, len(ttft_values) + 1), ttft_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Run Number')
        ax1.set_ylabel('TTFT (seconds)')
        ax1.set_title(f'TTFT Across Multiple Runs\nUser: {user_id}')
        ax1.grid(True, alpha=0.3)
        
        # Add mean line
        mean_ttft = np.mean(ttft_values)
        ax1.axhline(y=mean_ttft, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_ttft:.4f}s')
        ax1.legend()
        
        # Plot 2: TTFT distribution histogram
        ax2.hist(ttft_values, bins=max(3, len(ttft_values)//2), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=mean_ttft, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_ttft:.4f}s')
        ax2.set_xlabel('TTFT (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('TTFT Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = f"ttft_analysis_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {plot_filename}")
        
        plt.show()

    def print_detailed_analysis(self, user_id, ttft_values, prompt_length, stats):
        """Print detailed analysis results"""
        print(f"\n{'='*60}")
        print(f"TTFT ANALYSIS RESULTS FOR USER: {user_id}")
        print(f"{'='*60}")

        print(f"\nBASIC INFORMATION:")
        print(f"  User ID: {user_id}")
        print(f"  Prompt Length: {prompt_length} tokens")
        print(f"  Number of Runs: {len(ttft_values)}")
        print(f"  Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\nTTFT STATISTICS:")
        print(f"  Mean TTFT: {stats['mean']:.4f} seconds")
        print(f"  Median TTFT: {stats['median']:.4f} seconds")
        print(f"  Standard Deviation: {stats['std']:.4f} seconds")
        print(f"  Min TTFT: {stats['min']:.4f} seconds")
        print(f"  Max TTFT: {stats['max']:.4f} seconds")
        print(f"  25th Percentile: {stats['q25']:.4f} seconds")
        print(f"  75th Percentile: {stats['q75']:.4f} seconds")
        print(f"  Coefficient of Variation: {stats['cv']:.2f}%")

        print(f"\nPERFORMANCE INSIGHTS:")
        if stats['cv'] < 5:
            print(f"  Very consistent performance (CV < 5%)")
        elif stats['cv'] < 15:
            print(f"  Moderately consistent performance (CV < 15%)")
        else:
            print(f"  High variability in performance (CV >= 15%)")

        # Tokens per second calculation
        tokens_per_second = prompt_length / stats['mean']
        print(f"  Processing Speed: {tokens_per_second:.1f} tokens/second")

        print(f"\nRAW TTFT VALUES:")
        for i, ttft in enumerate(ttft_values, 1):
            print(f"  Run {i}: {ttft:.4f}s")

    def analyze_user_prompt(self, user_id, num_runs=5, show_plot=True, save_results=True):
        """Main method to analyze TTFT for a specific user prompt"""
        try:
            print(f"Starting TTFT analysis for user: {user_id}")
            
            # Load user data and create prompt
            history_data, candidate_data = self.load_user_data(user_id)
            prompt = self.create_recommendation_prompt(history_data, candidate_data)
            
            # Measure TTFT multiple times
            ttft_values, prompt_length = self.measure_ttft_multiple_runs(prompt, num_runs)
            
            # Analyze statistics
            stats = self.analyze_ttft_statistics(ttft_values)
            
            # Print detailed analysis
            self.print_detailed_analysis(user_id, ttft_values, prompt_length, stats)
            
            # Visualize results
            if show_plot:
                self.visualize_ttft_results(ttft_values, user_id, save_plot=True)
            
            # Save results to JSON
            if save_results:
                results = {
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'prompt_length': prompt_length,
                    'num_runs': num_runs,
                    'ttft_values': ttft_values,
                    'statistics': stats
                }
                
                results_filename = f"ttft_analysis_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_filename, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {results_filename}")
            
            return ttft_values, prompt_length, stats
            
        except Exception as e:
            print(f"Error analyzing user {user_id}: {e}")
            return None, None, None

def main():
    # Hardcoded user ID
    user_id = "reviewer_A103979529MRJY0U56QI4"
    num_runs = 5
    show_plot = True
    save_results = True

    # Initialize analyzer
    analyzer = TTFTAnalyzer()

    # Run analysis
    analyzer.analyze_user_prompt(
        user_id=user_id,
        num_runs=num_runs,
        show_plot=show_plot,
        save_results=save_results
    )

if __name__ == "__main__":
    main()
