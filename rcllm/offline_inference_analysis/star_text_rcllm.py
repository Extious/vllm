import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
import logging
import pandas as pd
from datetime import datetime
from star_text_prompt_tracker import TextPromptTracker

# --- Logging configuration ---
DEBUG_MODE = True  # True means debug mode, False means production mode

# Create logs directory if not exists
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplication
logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler (only in debug mode)
if DEBUG_MODE:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"star_text_rcllm_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Log file created: {log_file}")

# Visualization log file
visualization_log_file = os.path.join(log_dir, f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Initialize large model
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.8, enforce_eager=True, max_model_len=10000, dtype="half")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def load_copurchase_data():
    """Load copurchase data from CSV file"""
    copurchase_file = os.path.join(os.path.dirname(__file__), "../latency_analysis/copurchase_data.csv")
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

def save_visualization_data(prompt, input_ids, value_positions, query_position, tokenizer, user_id):
    """Save visualization data to file for debugging and analysis"""
    try:
        with open(visualization_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== VISUALIZATION DATA FOR USER {user_id} ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write prompt text
            f.write("=== FULL PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n")
            
            # Write tokenized information
            f.write("=== TOKEN ANALYSIS ===\n")
            f.write(f"Total tokens: {len(input_ids)}\n")
            f.write(f"Value positions count: {len(value_positions)}\n")
            f.write(f"Query position: {query_position}\n\n")
            
            # Write token-by-token mapping
            f.write("=== TOKEN-POSITION MAPPING ===\n")
            for i, token_id in enumerate(input_ids):
                token_text = tokenizer.decode([token_id]).replace('\n', '\\n').replace('\t', '\\t')
                position_type = "VALUE" if i in value_positions else "QUERY" if query_position and i in query_position else "NORMAL"
                f.write(f"Token {i:4d}: {token_id:6d} -> '{token_text}' [{position_type}]\n")
            
            f.write("\n=== VALUE POSITIONS DETAIL ===\n")
            sorted_value_positions = sorted(value_positions)
            for pos in sorted_value_positions:
                if pos < len(input_ids):
                    token_text = tokenizer.decode([input_ids[pos]]).replace('\n', '\\n').replace('\t', '\\t')
                    f.write(f"Value position {pos}: '{token_text}'\n")
            
            if query_position:
                f.write("\n=== QUERY POSITIONS DETAIL ===\n")
                for pos in query_position:
                    if pos < len(input_ids):
                        token_text = tokenizer.decode([input_ids[pos]]).replace('\n', '\\n').replace('\t', '\\t')
                        f.write(f"Query position {pos}: '{token_text}'\n")
            
            f.write("\n=== END OF VISUALIZATION DATA ===\n")
            
        logger.info(f"Visualization data saved to: {visualization_log_file}")
    except Exception as e:
        logger.error(f"Failed to save visualization data: {e}")

def load_user_data(user_id):
    """Load user history and candidates from reviewer_data_processed_strict.

    Reads a single reviewer file from the absolute dataset directory and splits
    the first 3 records as history and the rest as candidate items. Supports
    both "user_XXXX" and "reviewer_XXXX" style IDs.
    """
    base_dir = "/home/comp/24481750/rcllm/amazon/dataset/reviewer_data_processed_strict"

    # Normalize ID to on-disk filename convention
    if user_id.startswith("user_"):
        normalized_id = "reviewer_" + user_id[len("user_"):]
    elif user_id.startswith("reviewer_"):
        normalized_id = user_id
    else:
        normalized_id = f"reviewer_{user_id}"

    # Try multiple candidate file paths to be robust
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

def generate_recommendation_with_cache(user_id):
    # Load user data
    history_data, candidate_data = load_user_data(user_id)

    # Load copurchase data
    logger.info("Loading copurchase data...")
    copurchase_df = load_copurchase_data()
    
    # Create an instance of TextPromptTracker
    tracker = TextPromptTracker(tokenizer)

    logger.info(f"Number of loaded candidates: {len(candidate_data)}")
    logger.info(f"Example candidate: {json.dumps(candidate_data[0], indent=2)[:200]}...")

    # Create text completion format prompt
    full_prompt = tracker.create_text_completion_prompt(user_id, history_data, candidate_data, copurchase_df)
    
    # Track positions using text completion format
    all_chunk_ids, input_ids, value_positions, query_position = tracker.track_positions_from_text(
        full_prompt,
        candidate_data
    )
    
    # Save visualization data for debugging
    if DEBUG_MODE:
        save_visualization_data(full_prompt, input_ids, value_positions, query_position, tokenizer, user_id)

    # Verify that the last integer of query_position is within the range of input_ids
    if query_position:
        assert query_position[-1] < len(input_ids), \
            f"Query position end {query_position[-1]} out of bounds (input_ids length: {len(input_ids)})"

    # Calculate tracking statistics
    imp_indices = []
    imp_indices.extend(value_positions)

    logger.debug(f"Tracked token positions from candidate values: {len(imp_indices)} tokens, {imp_indices}")
    logger.info(f"Number of chunks in all_chunk_ids: {len(all_chunk_ids)}")

    # Initialize cache collection
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    cache_metadata['collect'] = False
    cache_metadata['check'] = False

    # Collect KV cache
    num_layer = 32  # Number of layers in Llama-3.1-8B
    chunk_past_key_values = []

    cache_metadata['collect'] = True

    # Collect KV cache for each block using text completion format
    for i in range(len(all_chunk_ids)):
        # Decode the chunk tokens to text
        prompt_text_for_kv_collection = tokenizer.decode(all_chunk_ids[i])
        
        # Use generate method for text completion
        llm.generate([prompt_text_for_kv_collection], SamplingParams(temperature=0.1, max_tokens=1))

        # Get KV cache from each layer of the model
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers

        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv

            if i == 0:
                # Handle first chunk
                temp_k = past_key_values[0].clone()
                temp_v = past_key_values[1].clone()
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                # Handle subsequent chunks, skip BOS token
                temp_k = past_key_values[0][1:].clone()
                temp_v = past_key_values[1][1:].clone()

                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_v), dim=0)

        logger.debug(f"Processed KV for chunk {i}, num_tokens in chunk (BOS-less): {len(all_chunk_ids[i])}, KV cache collected for {past_key_values[0].shape[0]} tokens (with BOS)")

    # Set merged KV cache
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
    logger.info(f"Final KV cache shape: {chunk_past_key_values[0][0].shape[0]}")

    # Generate using cache
    cache_metadata["check"] = True
    cache_metadata['collect'] = False
    cache_metadata['imp_indices'] = imp_indices

    print(f"input_ids_len: {len(input_ids)}")
    print(f"imp_indices_len: {len(cache_metadata['imp_indices'])}")
    print(f"First 10 imp_indices: {cache_metadata['imp_indices'][:10] if cache_metadata['imp_indices'] else 'None'}")

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    
    # Use generate method with text completion format
    output = llm.generate([full_prompt], sampling_params)

    print(f"Generation with cache: {output[0].outputs[0].text}")
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")

    cache_metadata["check"] = False
    cache_metadata['collect'] = False
    output = llm.generate([full_prompt], sampling_params)
    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")

if __name__ == "__main__":
    user_id = "user_A10FW892S59ABJ"
    logger.info("\n=====Generating Recommendations with Text Completion=====")
    generate_recommendation_with_cache(user_id)
    logger.info("===== End =====")
