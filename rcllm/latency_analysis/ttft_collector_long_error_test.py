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
import traceback
import sys

# Initialize model and tokenizer
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.8)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm.set_tokenizer(tokenizer)

def get_failed_user_ids(error_file="ttft_result_long_error.csv", max_users=5):
    """Get failed user IDs from the error CSV file (limited for testing)"""
    failed_users = []
    try:
        with open(error_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i >= max_users:  # Limit for testing
                    break
                failed_users.append(row['user_id'])
        print(f"Found {len(failed_users)} failed users in {error_file} (limited to {max_users} for testing)")
        return failed_users
    except FileNotFoundError:
        print(f"Error file {error_file} not found!")
        return []
    except Exception as e:
        print(f"Error reading error file: {e}")
        return []

def load_user_data_detailed(user_id):
    """Load user historical purchase data with detailed error analysis"""
    # Construct path to the reviewer data file
    data_file_path = os.path.join("/home/comp/24481750/rcllm/amazon/dataset/reviewer_data_processed", f"{user_id}.json")

    # Check if file exists
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Reviewer data file not found: {data_file_path}")

    # Read the complete data from the reviewer file
    with open(data_file_path, 'r') as f:
        all_data = json.load(f)

    # Detailed data validation
    data_info = {
        'total_items': len(all_data),
        'file_size': os.path.getsize(data_file_path),
        'file_path': data_file_path
    }

    # Ensure we have enough data
    if len(all_data) < 4:
        raise ValueError(f"Insufficient data for {user_id}: need at least 4 items, got {len(all_data)}")

    # Validate data structure
    for i, item in enumerate(all_data):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid data structure at index {i}: expected dict, got {type(item)}")
        
        # Check required fields
        required_fields = ['title', 'description', 'itemID', 'overall']
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            raise ValueError(f"Missing required fields at index {i}: {missing_fields}")

    # Split data: first 3 items as history, rest as candidates
    history_data = all_data[:3]
    candidate_data = all_data[3:]

    data_info.update({
        'history_count': len(history_data),
        'candidate_count': len(candidate_data),
        'history_items': [item.get('title', 'No title')[:50] + '...' for item in history_data],
        'candidate_items': [item.get('title', 'No title')[:50] + '...' for item in candidate_data]
    })

    return history_data, candidate_data, data_info

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

def measure_ttft_with_detailed_analysis(user_id, max_retries=2):
    """Measure TTFT with detailed error analysis and retry logic"""
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            print(f"  Attempt {retry_count + 1} for {user_id}...")
            
            # Load user data with detailed analysis
            history_data, candidate_data, data_info = load_user_data_detailed(user_id)
            print(f"    Data loaded: {data_info['total_items']} items, {data_info['history_count']} history, {data_info['candidate_count']} candidates")

            # Create the complete prompt
            input_prompt = create_recommendation_prompt(history_data, candidate_data)

            # Calculate prompt length in tokens
            prompt_tokens = tokenizer.encode(input_prompt)
            prompt_length = len(prompt_tokens)
            print(f"    Prompt length: {prompt_length} tokens")

            # Check if prompt is too long - skip this user if so
            if prompt_length > 120000:  # Leave some buffer for model context
                print(f"    SKIPPED: Prompt too long ({prompt_length} tokens)")
                return {
                    'success': False,
                    'ttft_full_prefill': None,
                    'prompt_length': prompt_length,
                    'data_info': data_info,
                    'retry_count': retry_count,
                    'error': f"Prompt too long: {prompt_length} tokens (max: 120000)",
                    'error_type': 'PromptTooLongError',
                    'skip_reason': 'prompt_too_long'
                }

            # Generate with full prefill (no cache)
            sampling_params = SamplingParams(temperature=0, max_tokens=256)
            output = llm.generate([input_prompt], sampling_params)

            # Calculate TTFT with full prefill
            ttft_full_prefill = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
            print(f"    TTFT: {ttft_full_prefill:.4f}s")

            # Return success with detailed info
            return {
                'success': True,
                'ttft_full_prefill': ttft_full_prefill,
                'prompt_length': prompt_length,
                'data_info': data_info,
                'retry_count': retry_count,
                'error': None,
                'skip_reason': None
            }

        except Exception as e:
            retry_count += 1
            last_error = str(e)
            error_traceback = traceback.format_exc()
            print(f"    Error (attempt {retry_count}): {last_error}")
            
            # If this is the last retry, return detailed error info
            if retry_count >= max_retries:
                return {
                    'success': False,
                    'ttft_full_prefill': None,
                    'prompt_length': None,
                    'data_info': None,
                    'retry_count': retry_count,
                    'error': last_error,
                    'error_traceback': error_traceback,
                    'error_type': type(e).__name__,
                    'skip_reason': 'error_after_retries'
                }
            
            # Wait before retry
            time.sleep(1)

def test_failed_users(failed_users, max_retries=2):
    """Test failed users with detailed analysis"""
    results = []
    successful_count = 0
    failed_count = 0
    skipped_count = 0

    print(f"\nTesting {len(failed_users)} failed users...")
    
    for i, user_id in enumerate(failed_users):
        print(f"\n[{i+1}/{len(failed_users)}] Testing {user_id}")
        
        # Measure TTFT with detailed analysis
        result = measure_ttft_with_detailed_analysis(user_id, max_retries)

        if result['success']:
            successful_count += 1
            print(f"  SUCCESS: TTFT = {result['ttft_full_prefill']:.4f}s")
        elif result.get('skip_reason') == 'prompt_too_long':
            skipped_count += 1
            print(f"  SKIPPED: Prompt too long ({result['prompt_length']} tokens)")
        else:
            failed_count += 1
            print(f"  FAILED: {result['error']}")
            print(f"    Error type: {result.get('error_type', 'Unknown')}")

        results.append(result)

    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"Total tested: {len(failed_users)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped (prompt too long): {skipped_count}")
    print(f"Success rate: {successful_count/len(failed_users)*100:.1f}%")

    # Detailed error analysis
    if failed_count > 0 or skipped_count > 0:
        print(f"\nError Analysis:")
        error_types = {}
        skip_reasons = {}
        for result in results:
            if not result['success']:
                error_type = result.get('error_type', 'Unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                skip_reason = result.get('skip_reason', 'unknown')
                skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
        
        if error_types:
            print(f"  Error types: {error_types}")
        if skip_reasons:
            print(f"  Skip reasons: {skip_reasons}")

    return results

if __name__ == "__main__":
    print("=== TTFT Error Retry Test Script ===")
    
    # Get failed user IDs from error file (limited for testing)
    failed_users = get_failed_user_ids("ttft_result_long_error.csv", max_users=5)
    
    if not failed_users:
        print("No failed users found in error file!")
        sys.exit(1)
    
    print(f"Testing users: {failed_users}")
    
    # Test failed users
    results = test_failed_users(failed_users, max_retries=2)
    
    print("\n=== Test Complete ===") 