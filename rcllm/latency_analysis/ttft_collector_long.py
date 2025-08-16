import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
import time
import csv
from datetime import datetime
import glob
from tqdm import tqdm

# Initialize model and tokenizer
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.8)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm.set_tokenizer(tokenizer)

def get_all_reviewer_ids():
    """Get all available reviewer IDs from the processed data directory"""
    data_dir = os.path.join(os.path.dirname(__file__), "../../../amazon/dataset/reviewer_data_processed")
    json_files = glob.glob(os.path.join(data_dir, "reviewer_*.json"))
    reviewer_ids = [os.path.basename(f).replace('.json', '') for f in json_files]
    return sorted(reviewer_ids)

def load_user_data(user_id):
    """Load user history and candidates from reviewer_data_processed_strict.

    Supports IDs in either "user_XXXX" or "reviewer_XXXX" format; reads a
    single reviewer JSON file and splits first 3 as history and the rest as
    candidates.
    """
    base_dir = "/home/comp/24481750/rcllm/amazon/dataset/reviewer_data_processed_strict"

    if user_id.startswith("user_"):
        normalized_id = "reviewer_" + user_id[len("user_"):]
    elif user_id.startswith("reviewer_"):
        normalized_id = user_id
    else:
        normalized_id = f"reviewer_{user_id}"

    candidate_paths = [
        os.path.join(base_dir, f"{normalized_id}.json"),
    ]
    if normalized_id.startswith("reviewer_"):
        raw_id = normalized_id[len("reviewer_"):]
        candidate_paths.append(os.path.join(base_dir, f"{raw_id}.json"))

    data_file_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            data_file_path = path
            break

    if data_file_path is None:
        raise FileNotFoundError(
            f"Reviewer data file not found. Tried: {', '.join(candidate_paths)}"
        )

    with open(data_file_path, 'r') as f:
        all_data = json.load(f)

    if len(all_data) < 4:
        raise ValueError(
            f"Insufficient data for {user_id}: need at least 4 items (3 history + 1 candidate), got {len(all_data)}"
        )

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

        # Generate with full prefill (no cache)
        sampling_params = SamplingParams(temperature=0, max_tokens=256)
        output = llm.generate([input_prompt], sampling_params)

        # Calculate TTFT with full prefill
        ttft_full_prefill = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time

        return ttft_full_prefill, prompt_length, None

    except Exception as e:
        # Return error information for logging
        return None, None, str(e)

def collect_ttft_data(user_ids, output_file="ttft_results.csv"):
    """Collect TTFT data for multiple users and save to CSV (one test per user)"""
    results = []
    successful_count = 0
    failed_count = 0
    failed_users = []

    # Create CSV file with headers
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'user_id', 'ttft_full_prefill', 'prompt_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Create error CSV file with headers
        error_file = "ttft_result_long_error.csv"
        with open(error_file, 'w', newline='') as error_csvfile:
            error_fieldnames = ['timestamp', 'user_id', 'error_message']
            error_writer = csv.DictWriter(error_csvfile, fieldnames=error_fieldnames)
            error_writer.writeheader()

            # Use tqdm for progress bar
            with tqdm(total=len(user_ids), desc="Processing users", unit="user") as pbar:
                for user_id in user_ids:
                    pbar.set_description(f"Processing {user_id}")
                    ttft_value, prompt_length, error_msg = measure_ttft_full_prefill(user_id)

                    if ttft_value is not None and prompt_length is not None:
                        result = {
                            'timestamp': datetime.now().isoformat(),
                            'user_id': user_id,
                            'ttft_full_prefill': ttft_value,
                            'prompt_length': prompt_length
                        }
                        results.append(result)
                        writer.writerow(result)
                        successful_count += 1

                        pbar.set_postfix({
                            'Success': successful_count,
                            'Failed': failed_count,
                            'TTFT': f"{ttft_value:.4f}s"
                        })
                    else:
                        failed_count += 1
                        failed_users.append(user_id)

                        # Write failed user to error CSV
                        error_result = {
                            'timestamp': datetime.now().isoformat(),
                            'user_id': user_id,
                            'error_message': error_msg if error_msg else 'Unknown error'
                        }
                        error_writer.writerow(error_result)

                        pbar.set_postfix({
                            'Success': successful_count,
                            'Failed': failed_count,
                            'Status': 'FAILED'
                        })

                    pbar.update(1)

    print(f"\n{'='*50}")
    print(f"Results saved to {output_file}")
    if failed_count > 0:
        print(f"Failed users saved to ttft_result_long_error.csv")
    print(f"Total processed: {len(user_ids)}, Successful: {successful_count}, Failed: {failed_count}")
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
    # user_ids = all_reviewer_ids[:200]  # Use first x reviewers

    # Option 3: Use all reviewers (uncomment to use - WARNING: this will take a long time)
    user_ids = all_reviewer_ids

    print("=== TTFT Collection Script ===")
    print(f"Collecting TTFT data for {len(user_ids)} user(s) (one test per user)")
    print(f"Selected users: {user_ids[:3]}{'...' if len(user_ids) > 3 else ''}")

    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"ttft_results_long_{timestamp}.csv"

    # Collect TTFT data (one test per user)
    results = collect_ttft_data(user_ids, output_file=output_filename)

    # Analyze results
    analyze_ttft_results(output_filename)

    print("\n=== Collection Complete ===")
