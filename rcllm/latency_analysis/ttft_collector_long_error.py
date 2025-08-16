import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

def get_failed_user_ids(error_file="ttft_result_long_error.csv"):
    """Get failed user IDs from the error CSV file"""
    failed_users = []
    try:
        with open(error_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                failed_users.append(row['user_id'])
        print(f"Found {len(failed_users)} failed users in {error_file}")
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

def measure_ttft_with_detailed_analysis(user_id, max_retries=3):
    """Measure TTFT with detailed error analysis and retry logic"""
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Load user data with detailed analysis
            history_data, candidate_data, data_info = load_user_data_detailed(user_id)

            # Create the complete prompt
            input_prompt = create_recommendation_prompt(history_data, candidate_data)

            # Calculate prompt length in tokens
            prompt_tokens = tokenizer.encode(input_prompt)
            prompt_length = len(prompt_tokens)

            # Check if prompt is too long - skip this user if so
            if prompt_length > 120000:  # Leave some buffer for model context
                return {
                    'success': False,
                    'ttft_full_prefill': None,
                    'prompt_length': prompt_length,
                    'data_info': data_info,
                    'retry_count': retry_count,
                    'error': f"Prompt too long: {prompt_length} tokens (max: 120000)",
                    'error_traceback': '',
                    'error_type': 'PromptTooLongError',
                    'skip_reason': 'prompt_too_long'
                }

            # Generate with full prefill (no cache)
            sampling_params = SamplingParams(temperature=0, max_tokens=256)
            output = llm.generate([input_prompt], sampling_params)

            # Calculate TTFT with full prefill
            ttft_full_prefill = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time

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

def analyze_error_patterns(error_results):
    """Analyze error patterns from failed tests"""
    error_counts = {}
    error_types = {}
    skip_reasons = {}
    
    for result in error_results:
        if not result['success']:
            error_msg = result['error']
            error_type = result.get('error_type', 'Unknown')
            skip_reason = result.get('skip_reason', 'unknown')
            
            # Count error types
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count skip reasons
            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            
            # Categorize errors
            if 'FileNotFoundError' in error_msg:
                error_counts['file_not_found'] = error_counts.get('file_not_found', 0) + 1
            elif 'Insufficient data' in error_msg:
                error_counts['insufficient_data'] = error_counts.get('insufficient_data', 0) + 1
            elif 'list index out of range' in error_msg:
                error_counts['index_error'] = error_counts.get('index_error', 0) + 1
            elif 'Invalid data structure' in error_msg:
                error_counts['invalid_structure'] = error_counts.get('invalid_structure', 0) + 1
            elif 'Missing required fields' in error_msg:
                error_counts['missing_fields'] = error_counts.get('missing_fields', 0) + 1
            elif 'Prompt too long' in error_msg:
                error_counts['prompt_too_long'] = error_counts.get('prompt_too_long', 0) + 1
            else:
                error_counts['other_errors'] = error_counts.get('other_errors', 0) + 1
    
    return {
        'error_counts': error_counts,
        'error_types': error_types,
        'skip_reasons': skip_reasons,
        'total_failures': len(error_results)
    }

def collect_ttft_data_for_failed_users(failed_users, output_file="ttft_results_retry.csv", max_retries=3):
    """Collect TTFT data for failed users with detailed error analysis"""
    results = []
    successful_count = 0
    failed_count = 0
    error_results = []
    prompt_too_long_users = []

    # Create CSV file with headers
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'user_id', 'ttft_full_prefill', 'prompt_length', 'retry_count', 'success', 'error_message', 'error_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Create detailed error CSV file
        detailed_error_file = "ttft_result_long_error_detailed.csv"
        with open(detailed_error_file, 'w', newline='') as error_csvfile:
            error_fieldnames = ['timestamp', 'user_id', 'error_message', 'error_type', 'error_traceback', 'retry_count', 'data_info']
            error_writer = csv.DictWriter(error_csvfile, fieldnames=error_fieldnames)
            error_writer.writeheader()

            # Create prompt too long CSV file
            prompt_too_long_file = "ttft_result_prompt_too_long.csv"
            with open(prompt_too_long_file, 'w', newline='') as prompt_csvfile:
                prompt_fieldnames = ['timestamp', 'user_id', 'prompt_length', 'total_items', 'history_count', 'candidate_count', 'file_size']
                prompt_writer = csv.DictWriter(prompt_csvfile, fieldnames=prompt_fieldnames)
                prompt_writer.writeheader()

                # Use tqdm for progress bar
                with tqdm(total=len(failed_users), desc="Retesting failed users", unit="user") as pbar:
                    for user_id in failed_users:
                        pbar.set_description(f"Retesting {user_id}")
                        
                        # Measure TTFT with detailed analysis
                        result = measure_ttft_with_detailed_analysis(user_id, max_retries)

                        if result['success']:
                            # Write successful result
                            csv_result = {
                                'timestamp': datetime.now().isoformat(),
                                'user_id': user_id,
                                'ttft_full_prefill': result['ttft_full_prefill'],
                                'prompt_length': result['prompt_length'],
                                'retry_count': result['retry_count'],
                                'success': True,
                                'error_message': '',
                                'error_type': ''
                            }
                            results.append(csv_result)
                            writer.writerow(csv_result)
                            successful_count += 1

                            pbar.set_postfix({
                                'Success': successful_count,
                                'Failed': failed_count,
                                'TTFT': f"{result['ttft_full_prefill']:.4f}s",
                                'Retries': result['retry_count']
                            })
                        else:
                            # Check if it's a prompt too long error
                            if result.get('skip_reason') == 'prompt_too_long':
                                # Write to prompt too long CSV
                                prompt_result = {
                                    'timestamp': datetime.now().isoformat(),
                                    'user_id': user_id,
                                    'prompt_length': result['prompt_length'],
                                    'total_items': result['data_info']['total_items'] if result['data_info'] else 0,
                                    'history_count': result['data_info']['history_count'] if result['data_info'] else 0,
                                    'candidate_count': result['data_info']['candidate_count'] if result['data_info'] else 0,
                                    'file_size': result['data_info']['file_size'] if result['data_info'] else 0
                                }
                                prompt_writer.writerow(prompt_result)
                                prompt_too_long_users.append(user_id)
                            else:
                                # Write failed result with detailed error info
                                failed_count += 1
                                error_results.append(result)

                                csv_result = {
                                    'timestamp': datetime.now().isoformat(),
                                    'user_id': user_id,
                                    'ttft_full_prefill': None,
                                    'prompt_length': None,
                                    'retry_count': result['retry_count'],
                                    'success': False,
                                    'error_message': result['error'],
                                    'error_type': result.get('error_type', 'Unknown')
                                }
                                writer.writerow(csv_result)

                                # Write detailed error info
                                detailed_error_result = {
                                    'timestamp': datetime.now().isoformat(),
                                    'user_id': user_id,
                                    'error_message': result['error'],
                                    'error_type': result.get('error_type', 'Unknown'),
                                    'error_traceback': result.get('error_traceback', ''),
                                    'retry_count': result['retry_count'],
                                    'data_info': json.dumps(result.get('data_info', {}))
                                }
                                error_writer.writerow(detailed_error_result)

                            pbar.set_postfix({
                                'Success': successful_count,
                                'Failed': failed_count,
                                'Skipped': len(prompt_too_long_users),
                                'Error': result.get('error_type', 'Unknown')
                            })

                        pbar.update(1)

    # Analyze error patterns
    error_analysis = analyze_error_patterns(error_results)

    print(f"\n{'='*60}")
    print(f"Retry Results Summary:")
    print(f"Results saved to {output_file}")
    print(f"Detailed errors saved to {detailed_error_file}")
    print(f"Prompt too long users saved to {prompt_too_long_file}")
    print(f"Total processed: {len(failed_users)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped (prompt too long): {len(prompt_too_long_users)}")
    print(f"Success rate: {successful_count/len(failed_users)*100:.1f}%")
    
    if error_analysis['total_failures'] > 0:
        print(f"\nError Analysis:")
        print(f"Total failures: {error_analysis['total_failures']}")
        print(f"Error types: {error_analysis['error_types']}")
        print(f"Error categories: {error_analysis['error_counts']}")
        print(f"Skip reasons: {error_analysis['skip_reasons']}")

    return results, error_analysis

def generate_error_report(error_analysis, output_file="error_analysis_report.txt"):
    """Generate a detailed error analysis report"""
    with open(output_file, 'w') as f:
        f.write("TTFT Error Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("Error Summary:\n")
        f.write(f"Total failures: {error_analysis['total_failures']}\n\n")
        
        f.write("Error Types:\n")
        for error_type, count in error_analysis['error_types'].items():
            f.write(f"  {error_type}: {count}\n")
        f.write("\n")
        
        f.write("Error Categories:\n")
        for category, count in error_analysis['error_counts'].items():
            f.write(f"  {category}: {count}\n")
        f.write("\n")
        
        f.write("Recommendations:\n")
        if error_analysis['error_counts'].get('file_not_found', 0) > 0:
            f.write("- Check data file paths and ensure all reviewer files exist\n")
        if error_analysis['error_counts'].get('insufficient_data', 0) > 0:
            f.write("- Filter users with insufficient data (less than 4 items)\n")
        if error_analysis['error_counts'].get('index_error', 0) > 0:
            f.write("- Add bounds checking for data access\n")
        if error_analysis['error_counts'].get('invalid_structure', 0) > 0:
            f.write("- Validate data structure before processing\n")
        if error_analysis['error_counts'].get('missing_fields', 0) > 0:
            f.write("- Handle missing required fields gracefully\n")
        if error_analysis['error_counts'].get('prompt_too_long', 0) > 0:
            f.write("- Implement prompt truncation or chunking\n")
    
    print(f"Error analysis report saved to {output_file}")

if __name__ == "__main__":
    print("=== TTFT Error Retry and Analysis Script ===")
    
    # Get failed user IDs from error file
    failed_users = get_failed_user_ids("ttft_result_long_error.csv")
    
    if not failed_users:
        print("No failed users found in error file!")
        sys.exit(1)
    
    print(f"Found {len(failed_users)} failed users to retry")
    print(f"Sample failed users: {failed_users[:5]}")
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"ttft_results_retry_{timestamp}.csv"
    
    # Collect TTFT data for failed users with detailed analysis
    results, error_analysis = collect_ttft_data_for_failed_users(
        failed_users, 
        output_file=output_filename,
        max_retries=3
    )
    
    # Generate error analysis report
    generate_error_report(error_analysis, f"error_analysis_report_{timestamp}.txt")
    
    print("\n=== Retry and Analysis Complete ===") 