import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
import time
import csv
from datetime import datetime
import glob
import pandas as pd

# Initialize model and tokenizer
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.5)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm.set_tokenizer(tokenizer)

def get_all_reviewer_ids():
    """Get all available reviewer IDs from the processed data directory"""
    data_dir = os.path.join(os.path.dirname(__file__), "../../../amazon/dataset/reviewer_data_processed_strict")
    json_files = glob.glob(os.path.join(data_dir, "reviewer_*.json"))
    reviewer_ids = [os.path.basename(f).replace('.json', '') for f in json_files]
    return sorted(reviewer_ids)

def load_copurchase_data():
    """Load copurchase data from CSV file"""
    copurchase_file = os.path.join(os.path.dirname(__file__), "copurchase_data.csv")
    copurchase_df = pd.read_csv(copurchase_file)
    return copurchase_df

def get_copurchase_counts(copurchase_df, candidate_item, history_items):
    """Get copurchase counts between candidate item and each history item"""
    copurchase_info = {}
    
    candidate_item_id = candidate_item.get('itemID', '')
    
    for i, history_item in enumerate(history_items):
        history_item_id = history_item.get('itemID', '')
        
        # Check both directions (item1->item2 and item2->item1)
        count1 = copurchase_df[
            (copurchase_df['item1'] == candidate_item_id) & 
            (copurchase_df['item2'] == history_item_id)
        ]['copurchase_count'].sum()
        
        count2 = copurchase_df[
            (copurchase_df['item1'] == history_item_id) & 
            (copurchase_df['item2'] == candidate_item_id)
        ]['copurchase_count'].sum()
        
        total_count = int(count1 + count2)
        copurchase_info[f"Number of users who bought both this item and Item ID {i+1}"] = total_count
    
    return copurchase_info

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

def create_diagram_format_conversation(user_id, history_data, candidate_data, copurchase_df):
    """Create conversation following the exact format from the diagram using JSON format"""
    
    # System introduction
    system_intro = "You are an intelligent assistant that can rank items based on the user's preference."
    
    # User provides purchase history with native JSON data and task description
    user_history_intro = f"User {user_id} has purchased the following items in this order:"
    
    history_items = []
    for i, item in enumerate(history_data):
        history_items.append(json.dumps(item, indent=2))
    
    candidate_count = len(candidate_data)
    task_description = f"I will provide you with {candidate_count} items, each indicated by number identifier []. Analyze the user's purchase history to identify preferences and purchase patterns. Then, rank the candidate items based on their alignment with the user's preferences and other contextual factors."
    
    history_content = f"{user_history_intro}\n" + "\n\n".join(history_items) + f"\n\n{task_description}"
    
    # Assistant acknowledgment
    assistant_ack = "Okay, please provide the items."
    
    # User provides candidate items with native JSON data and copurchase information
    candidate_items = []
    for i, item in enumerate(candidate_data):
        # Get copurchase counts for this candidate item with each history item
        copurchase_info = get_copurchase_counts(copurchase_df, item, history_data)
        
        # Merge copurchase information into the item JSON
        item_with_copurchase = item.copy()
        item_with_copurchase.update(copurchase_info)
        
        # Create candidate item text with merged JSON data
        candidate_item_text = f"[{i+1}]\n{json.dumps(item_with_copurchase, indent=2)}"
        candidate_items.append(candidate_item_text)
    
    # Assistant acknowledges each item
    assistant_acks = []
    for i in range(len(candidate_data)):
        assistant_acks.append(f"Received item [{i+1}].")
    
    # Final user instructions for ranking with dynamic candidate count
    final_instructions = f"""Analyze the user's purchase history to identify user preferences and purchase patterns.
Then, rank the {candidate_count} items above based on their alignment with the user's preferences and other contextual factors.
All the items should be included and listed using identifiers, in descending order of the user's preference.
The most preferred recommendation item should be listed first.
The output format should be [] > [], where each [] is an identifier, e.g., [1] > [2].
Only respond with the ranking results, do not say any word or explain.
Output in the following JSON format: {{ "rank": "[] > [].. > []" }}"""
    
    # Construct the conversation using JSON format
    conversation = [
        {
            "role": "system",
            "content": system_intro
        },
        {
            "role": "user",
            "content": history_content
        },
        {
            "role": "assistant",
            "content": assistant_ack
        }
    ]
    
    # Add candidate items and acknowledgments
    for i in range(len(candidate_data)):
        conversation.append({
            "role": "user",
            "content": candidate_items[i]
        })
        conversation.append({
            "role": "assistant",
            "content": assistant_acks[i]
        })
    
    # Add final instructions
    conversation.append({
        "role": "user",
        "content": final_instructions
    })
    
    return conversation

def measure_ttft_diagram_format(user_id, copurchase_df):
    """Measure TTFT with diagram format conversation for a given user"""
    try:
        # Load user data
        history_data, candidate_data = load_user_data(user_id)

        # Create the complete conversation using diagram format
        conversation = create_diagram_format_conversation(user_id, history_data, candidate_data, copurchase_df)

        # Calculate prompt length in tokens directly from conversation
        prompt_tokens = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False)
        prompt_length = len(prompt_tokens)

        print(f"Processing user {user_id}")
        print(f"Prompt length: {prompt_length} tokens")
        print(f"History items: {len(history_data)}, Candidate items: {len(candidate_data)}")

        # Generate with chat method
        sampling_params = SamplingParams(temperature=0, max_tokens=256)
        output = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)

        # Calculate TTFT
        ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
        
        # Calculate TTLF
        ttlf = output[0].metrics.finished_time - output[0].metrics.first_scheduled_time

        print(f"TTFT: {ttft}")
        print(f"TTLF: {ttlf}")

        return ttft, ttlf, prompt_length

    except Exception as e:
        print(f"Error processing user {user_id}: {e}")
        return None, None, None

def collect_metrics_data(user_ids, output_file="latency_results_diagram_format.csv"):
    """Collect TTFT data for multiple users using diagram format prompt"""
    results = []

    # Load copurchase data
    print("Loading copurchase data...")
    copurchase_df = load_copurchase_data()
    print(f"Loaded copurchase data with {len(copurchase_df)} records")

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'user_id', 'ttft', 'ttlf', 'prompt_length', 'history_items', 'candidate_items', 'total_items']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for user_id in user_ids:
            print(f"\n{'='*50}")
            ttft_value, ttlf_value, prompt_length = measure_ttft_diagram_format(user_id, copurchase_df)

            if ttft_value is not None and prompt_length is not None:
                # Load user data to get item counts
                history_data, candidate_data = load_user_data(user_id)
                history_count = len(history_data)
                candidate_count = len(candidate_data)
                total_count = history_count + candidate_count
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'user_id': user_id,
                    'ttft': ttft_value,
                    'ttlf': ttlf_value,
                    'prompt_length': prompt_length,
                    'history_items': history_count,
                    'candidate_items': candidate_count,
                    'total_items': total_count
                }
                results.append(result)
                writer.writerow(result)

                print(f"[SUCCESS] User {user_id}: TTFT = {ttft_value:.4f}s, TTLF = {ttlf_value:.4f}s, Prompt Length = {prompt_length} tokens")
            else:
                print(f"[FAILED] Failed to process user {user_id}")

    print(f"\n{'='*50}")
    print(f"Results saved to {output_file}")
    return results

def analyze_latency_results(results_file="latency_results_diagram_format.csv"):
    """Analyze the collected TTFT results"""
    try:
        import pandas as pd
        df = pd.read_csv(results_file)

        print(f"\n=== TTFT Analysis ===")
        print(f"Total measurements: {len(df)}")
        print(f"Number of users: {df['user_id'].nunique()}")
        print(f"Average TTFT: {df['ttft'].mean():.4f}s")
        print(f"Median TTFT: {df['ttft'].median():.4f}s")
        print(f"Min TTFT: {df['ttft'].min():.4f}s")
        print(f"Max TTFT: {df['ttft'].max():.4f}s")
        print(f"Standard deviation: {df['ttft'].std():.4f}s")

        print(f"\n=== Prompt Length Analysis ===")
        print(f"Average prompt length: {df['prompt_length'].mean():.1f} tokens")
        print(f"Median prompt length: {df['prompt_length'].median():.1f} tokens")
        print(f"Min prompt length: {df['prompt_length'].min()} tokens")
        print(f"Max prompt length: {df['prompt_length'].max()} tokens")
        print(f"Prompt length std: {df['prompt_length'].std():.1f} tokens")

        print(f"\n=== Per-User Statistics ===")
        user_stats = df[['user_id', 'ttft', 'prompt_length']]
        print(user_stats.to_string(index=False))

    except ImportError:
        print("pandas not available for analysis. Raw data saved in CSV format.")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    # Get all available reviewer IDs
    all_reviewer_ids = get_all_reviewer_ids()
    print(f"Found {len(all_reviewer_ids)} reviewer data files")

    # Use first N reviewers for testing
    user_ids = all_reviewer_ids[:3000]  # Use first 100 reviewers

    print("=== Diagram Format Latency Collection Script ===")
    print(f"Collecting latency data for {len(user_ids)} user(s) using diagram format prompt")
    print(f"Selected users: {user_ids[:3]}{'...' if len(user_ids) > 3 else ''}")

    # Collect latency data
    results = collect_metrics_data(user_ids, output_file="latency_results_diagram_format.csv")

    # Analyze results
    analyze_latency_results("latency_results_diagram_format.csv")  

    print("\n=== Collection Complete ===") 