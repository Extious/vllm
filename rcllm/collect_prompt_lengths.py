#!/usr/bin/env python3
"""
Script to collect prompt token lengths for each user without accessing vLLM.
This script loads user data and calculates the token length of the generated prompts.
"""

import os
import json
import glob
import csv
from datetime import datetime
from transformers import AutoTokenizer

def get_all_reviewer_ids():
    """Get all available reviewer IDs from the processed data directory"""
    data_dir = os.path.join(os.path.dirname(__file__), "../../../amazon/dataset/reviewer_data_processed")
    json_files = glob.glob(os.path.join(data_dir, "reviewer_*.json"))
    reviewer_ids = [os.path.basename(f).replace('.json', '') for f in json_files]
    return sorted(reviewer_ids)

def load_user_data(user_id):
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

def create_recommendation_prompt(history_data, candidate_data):
    """Create the recommendation prompt (same as in ttft_collector.py)"""
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

def calculate_prompt_length(user_id, tokenizer):
    """Calculate prompt token length for a given user"""
    try:
        # Load user data
        history_data, candidate_data = load_user_data(user_id)

        # Create the complete prompt
        input_prompt = create_recommendation_prompt(history_data, candidate_data)

        # Calculate prompt length in tokens
        prompt_tokens = tokenizer.encode(input_prompt)
        prompt_length = len(prompt_tokens)

        return {
            'user_id': user_id,
            'prompt_length': prompt_length,
            'history_items': len(history_data),
            'candidate_items': len(candidate_data),
            'total_items': len(history_data) + len(candidate_data),
            'prompt_chars': len(input_prompt),
            'status': 'success'
        }

    except Exception as e:
        return {
            'user_id': user_id,
            'prompt_length': None,
            'history_items': None,
            'candidate_items': None,
            'total_items': None,
            'prompt_chars': None,
            'status': f'error: {str(e)}'
        }

def collect_prompt_lengths(user_ids, output_file="prompt_lengths.csv"):
    """Collect prompt lengths for multiple users and save to CSV"""
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print("Tokenizer loaded successfully")

    results = []

    # Create CSV file with headers
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'user_id', 'prompt_length', 'history_items', 'candidate_items', 'total_items', 'prompt_chars', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Processing {len(user_ids)} users...")

        for i, user_id in enumerate(user_ids, 1):
            if i % 100 == 0 or i == 1:
                print(f"Processing user {i}/{len(user_ids)}: {user_id}")

            result = calculate_prompt_length(user_id, tokenizer)
            result['timestamp'] = datetime.now().isoformat()

            results.append(result)
            writer.writerow(result)

            if result['status'] == 'success':
                if i % 100 == 0 or i <= 10:  # Show details for first 10 and every 100th
                    print(f"YES: {user_id}: {result['prompt_length']} tokens ({result['history_items']} history + {result['candidate_items']} candidates)")
            else:
                print(f"NO: {user_id}: {result['status']}")

    print(f"\nResults saved to {output_file}")
    return results

def analyze_prompt_lengths(results_file="prompt_lengths.csv"):
    """Analyze the collected prompt length results"""
    try:
        import pandas as pd
        df = pd.read_csv(results_file)
        
        # Filter successful results
        success_df = df[df['status'] == 'success']
        failed_df = df[df['status'] != 'success']
        
        print(f"\n=== Prompt Length Analysis ===")
        print(f"Total users processed: {len(df)}")
        print(f"Successful: {len(success_df)}")
        print(f"Failed: {len(failed_df)}")
        
        if len(success_df) > 0:
            print(f"\n=== Prompt Length Statistics ===")
            print(f"Average prompt length: {success_df['prompt_length'].mean():.1f} tokens")
            print(f"Median prompt length: {success_df['prompt_length'].median():.1f} tokens")
            print(f"Min prompt length: {success_df['prompt_length'].min()} tokens")
            print(f"Max prompt length: {success_df['prompt_length'].max()} tokens")
            print(f"Standard deviation: {success_df['prompt_length'].std():.1f} tokens")
            
            print(f"\n=== Data Size Statistics ===")
            print(f"Average history items: {success_df['history_items'].mean():.1f}")
            print(f"Average candidate items: {success_df['candidate_items'].mean():.1f}")
            print(f"Average total items: {success_df['total_items'].mean():.1f}")
            
            # Show distribution
            print(f"\n=== Prompt Length Distribution ===")
            bins = [0, 1000, 2000, 3000, 4000, 5000, 10000, float('inf')]
            labels = ['<1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K', '5K-10K', '>10K']
            
            for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                if bin_end == float('inf'):
                    count = len(success_df[success_df['prompt_length'] >= bin_start])
                else:
                    count = len(success_df[(success_df['prompt_length'] >= bin_start) & (success_df['prompt_length'] < bin_end)])
                percentage = count / len(success_df) * 100
                print(f"{labels[i]}: {count} users ({percentage:.1f}%)")
        
        if len(failed_df) > 0:
            print(f"\n=== Failed Cases ===")
            failure_reasons = failed_df['status'].value_counts()
            for reason, count in failure_reasons.items():
                print(f"{reason}: {count} users")

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
    # user_ids = all_reviewer_ids[:100]  # Process first 100 reviewers

    # Option 3: Use all reviewers (WARNING: this will take a long time)
    user_ids = all_reviewer_ids

    print("=== Prompt Length Collection Script ===")
    print(f"Collecting prompt lengths for {len(user_ids)} user(s)")
    print(f"Selected users: {user_ids[:3]}{'...' if len(user_ids) > 3 else ''}")

    # Collect prompt lengths
    results = collect_prompt_lengths(user_ids, output_file="prompt_lengths.csv")

    # Analyze results
    analyze_prompt_lengths("prompt_lengths.csv")

    print("\n=== Collection Complete ===")
