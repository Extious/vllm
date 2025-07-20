import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
import time
import csv
from datetime import datetime
import glob

# Initialize model and tokenizer
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.5)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm.set_tokenizer(tokenizer)

def get_all_reviewer_ids():
    """Get all available reviewer IDs from the processed data directory"""
    data_dir = os.path.join(os.path.dirname(__file__), "../../../amazon/dataset/reviewer_data_processed_cleaned")
    json_files = glob.glob(os.path.join(data_dir, "reviewer_*.json"))
    reviewer_ids = [os.path.basename(f).replace('.json', '') for f in json_files]
    return sorted(reviewer_ids)

def load_user_data(user_id):
    """Load user historical purchase data and candidates from reviewer data files"""
    # Construct path to the reviewer data file
    data_file_path = os.path.join(os.path.dirname(__file__), "../../../amazon/dataset/reviewer_data_processed_cleaned", f"{user_id}.json")

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

def create_recommendation_prompt(history_data, candidate_data):
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

def measure_ttft_full_prefill(user_id):
    """Measure TTFT with full prefill for a given user (single run)"""
    try:
        # Load user data
        history_data, candidate_data = load_user_data(user_id)

        # Create the complete prompt
        input_prompt = create_recommendation_prompt(history_data, candidate_data)

        # Calculate prompt length in tokens
        prompt_tokens = tokenizer.encode(input_prompt)
        prompt_length = len(prompt_tokens)

        print(f"Processing user {user_id}")
        print(f"Prompt length: {prompt_length} tokens")

        # Generate with full prefill (no cache)
        sampling_params = SamplingParams(temperature=0, max_tokens=256)
        output = llm.generate([input_prompt], sampling_params)

        # Calculate TTFT with full prefill
        ttft_full_prefill = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time

        print(f"TTFT with full prefill: {ttft_full_prefill}")

        return ttft_full_prefill, prompt_length

    except Exception as e:
        print(f"Error processing user {user_id}: {e}")
        return None, None

def collect_ttft_data(user_ids, output_file="ttft_results.csv"):
    """Collect TTFT data for multiple users and save to CSV (one test per user)"""
    results = []

    # Create CSV file with headers
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'user_id', 'ttft_full_prefill', 'prompt_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for user_id in user_ids:
            print(f"\n{'='*50}")
            ttft_value, prompt_length = measure_ttft_full_prefill(user_id)

            if ttft_value is not None and prompt_length is not None:
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'user_id': user_id,
                    'ttft_full_prefill': ttft_value,
                    'prompt_length': prompt_length
                }
                results.append(result)
                writer.writerow(result)

                print(f"[SUCCESS] User {user_id}: TTFT = {ttft_value:.4f}s, Prompt Length = {prompt_length} tokens")
            else:
                print(f"[FAILED] Failed to process user {user_id}")

    print(f"\n{'='*50}")
    print(f"Results saved to {output_file}")
    return results

def analyze_ttft_results(results_file="ttft_results.csv"):
    """Analyze the collected TTFT results"""
    try:
        import pandas as pd
        df = pd.read_csv(results_file)

        print(f"\n=== TTFT Analysis ===")
        print(f"Total measurements: {len(df)}")
        print(f"Number of users: {df['user_id'].nunique()}")
        print(f"Average TTFT: {df['ttft_full_prefill'].mean():.4f}s")
        print(f"Median TTFT: {df['ttft_full_prefill'].median():.4f}s")
        print(f"Min TTFT: {df['ttft_full_prefill'].min():.4f}s")
        print(f"Max TTFT: {df['ttft_full_prefill'].max():.4f}s")
        print(f"Standard deviation: {df['ttft_full_prefill'].std():.4f}s")

        print(f"\n=== Prompt Length Analysis ===")
        print(f"Average prompt length: {df['prompt_length'].mean():.1f} tokens")
        print(f"Median prompt length: {df['prompt_length'].median():.1f} tokens")
        print(f"Min prompt length: {df['prompt_length'].min()} tokens")
        print(f"Max prompt length: {df['prompt_length'].max()} tokens")
        print(f"Prompt length std: {df['prompt_length'].std():.1f} tokens")

        # Per-user statistics
        print(f"\n=== Per-User Statistics ===")
        user_stats = df[['user_id', 'ttft_full_prefill', 'prompt_length']]
        print(user_stats.to_string(index=False))

    except ImportError:
        print("pandas not available for analysis. Raw data saved in CSV format.")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    # Get all available reviewer IDs
    all_reviewer_ids = get_all_reviewer_ids()
    print(f"Found {len(all_reviewer_ids)} reviewer data files")

    # Example usage - you can choose to use single reviewer or multiple reviewers
    # Option 1: Use a single reviewer
    # user_ids = ["reviewer_A1A0PPF8HE508X"]

    # Option 2: Use first N reviewers (uncomment to use)
    user_ids = all_reviewer_ids[:200]  # Use first 5 reviewers

    # Option 3: Use all reviewers (uncomment to use - WARNING: this will take a long time)
    # user_ids = all_reviewer_ids

    print("=== TTFT Collection Script ===")
    print(f"Collecting TTFT data for {len(user_ids)} user(s) (one test per user)")
    print(f"Selected users: {user_ids[:3]}{'...' if len(user_ids) > 3 else ''}")

    # Collect TTFT data (one test per user)
    results = collect_ttft_data(user_ids, output_file="ttft_results_short.csv")

    # Analyze results
    analyze_ttft_results("ttft_results_short.csv")

    print("\n=== Collection Complete ===")
