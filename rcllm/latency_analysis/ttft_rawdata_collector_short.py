#!/usr/bin/env python3
"""
Integrated script to collect data statistics and prompt lengths for Amazon reviewer data.
Combines functionality from collect_prompt_lengths.py and data_stats_collector.py.
"""

import os
import json
import glob
import csv
from datetime import datetime
from transformers import AutoTokenizer

def get_all_reviewer_ids():
    """Get all available reviewer IDs from the processed data directory"""
    data_dir = os.path.join(os.path.dirname(__file__), "../../../amazon/dataset/reviewer_data_processed_cleaned")
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

def collect_user_stats(user_id, tokenizer):
    """Collect statistics for a given user"""
    try:
        history_data, candidate_data = load_user_data(user_id)
        input_prompt = create_recommendation_prompt(history_data, candidate_data)

        # Calculate prompt length in tokens
        prompt_tokens = tokenizer.encode(input_prompt)

        return {
            'user_id': user_id,
            'total_items': len(history_data) + len(candidate_data),
            'prompt_length_tokens': len(prompt_tokens)
        }
    except Exception as e:
        print(f"Error processing user {user_id}: {e}")
        return None

def collect_all_stats(user_ids, output_file):
    """Collect statistics for multiple users and save to CSV"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print("Tokenizer loaded successfully")

    results = []

    # Ensure output file is in rcllm directory
    output_path = os.path.join(os.path.dirname(__file__), output_file)

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['user_id', 'total_items', 'prompt_length_tokens']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Processing {len(user_ids)} users...")

        for i, user_id in enumerate(user_ids, 1):
            if i % 100 == 0 or i <= 10:
                print(f"Processing user {i}/{len(user_ids)}: {user_id}")

            result = collect_user_stats(user_id, tokenizer)

            if result is not None:
                results.append(result)
                writer.writerow(result)

                if i % 100 == 0 or i <= 10:
                    print(f"SUCCESS: {user_id}: {result['prompt_length_tokens']} tokens "
                          f"({result['total_items']} total items)")
            else:
                print(f"FAILED: {user_id}")

    print(f"\nResults saved to {output_path}")
    return results

def analyze_stats(results_file):
    """Analyze the collected statistics"""
    try:
        import pandas as pd
        df = pd.read_csv(results_file)

        print(f"\n=== Analysis Results ===")
        print(f"Total users processed: {len(df)}")

        print(f"\n=== Data Statistics ===")
        print(f"Average total items: {df['total_items'].mean():.1f}")
        print(f"Min total items: {df['total_items'].min()}")
        print(f"Max total items: {df['total_items'].max()}")

        print(f"\n=== Prompt Length Statistics ===")
        print(f"Average prompt length: {df['prompt_length_tokens'].mean():.1f} tokens")
        print(f"Median prompt length: {df['prompt_length_tokens'].median():.1f} tokens")
        print(f"Min prompt length: {df['prompt_length_tokens'].min()} tokens")
        print(f"Max prompt length: {df['prompt_length_tokens'].max()} tokens")
        print(f"Standard deviation: {df['prompt_length_tokens'].std():.1f} tokens")

        # Distribution analysis
        print(f"\n=== Prompt Length Distribution ===")
        bins = [0, 1000, 2000, 3000, 4000, 5000, 10000, float('inf')]
        labels = ['<1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K', '5K-10K', '>10K']

        for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            if bin_end == float('inf'):
                count = len(df[df['prompt_length_tokens'] >= bin_start])
            else:
                count = len(df[(df['prompt_length_tokens'] >= bin_start) &
                             (df['prompt_length_tokens'] < bin_end)])
            percentage = count / len(df) * 100
            print(f"{labels[i]}: {count} users ({percentage:.1f}%)")

    except ImportError:
        print("pandas not available for detailed analysis. Raw data saved in CSV format.")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    # Get all available reviewer IDs
    all_reviewer_ids = get_all_reviewer_ids()
    print(f"Found {len(all_reviewer_ids)} reviewer data files")

    # Configuration options
    # Option 1: Use a small sample for testing
    # user_ids = all_reviewer_ids[:10]

    # Option 2: Use first N reviewers (recommended for batch processing)
    # user_ids = all_reviewer_ids[:100]

    # Option 3: Use all reviewers
    user_ids = all_reviewer_ids

    print("=== Data Statistics Collection Script ===")
    print(f"Collecting statistics for {len(user_ids)} user(s)")
    print(f"Selected users: {user_ids[:3]}{'...' if len(user_ids) > 3 else ''}")

    # Collect statistics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ttft_rawdata_short_{timestamp}.csv"
    results = collect_all_stats(user_ids, output_file)

    # Analyze results
    analyze_stats(os.path.join(os.path.dirname(__file__), output_file))

    print("\n=== Collection Complete ===")
