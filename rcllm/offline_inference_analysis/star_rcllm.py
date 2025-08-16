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
from star_prompt_tracker import PromptFieldTracker

# --- Logging configuration ---
DEBUG_MODE = True  # True means debug mode, False means production mode
logger = logging.getLogger(__name__)
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# Initialize large model
# llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.5, enforce_eager=True, max_model_len=10000)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# llm.set_tokenizer(tokenizer) # set_tokenizer is deprecated

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
        candidate_item_text = f"[{i}]\n{json.dumps(item_with_copurchase, indent=2)}"
        candidate_items.append(candidate_item_text)
    
    # Assistant acknowledges each item
    assistant_acks = []
    for i in range(len(candidate_data)):
        assistant_acks.append(f"Received item [{i}].")
    
    # Final user instructions for ranking with dynamic candidate count
    final_instructions = f"""Analyze the user's purchase history to identify user preferences and purchase patterns.
Then, rank the {candidate_count} items above based on their alignment with the user's preferences and other contextual factors.
All the items should be included and listed using identifiers, in descending order of the user's preference.
The most preferred recommendation item should be listed first.
The output format should be [] > [], where each [] is an identifier, e.g., [0] > [1].
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

def generate_recommendation_with_cache(user_id):
    # Load user data
    history_data, candidate_data = load_user_data(user_id)

    # Load copurchase data
    logger.info("Loading copurchase data...")
    copurchase_df = load_copurchase_data()
    
    # Create the complete conversation using diagram format
    conversation = create_diagram_format_conversation(user_id, history_data, candidate_data, copurchase_df)
    
    # Create an instance of PromptFieldTracker
    tracker = PromptFieldTracker(tokenizer)

    logger.info(f"Number of loaded candidates: {len(candidate_data)}")
    logger.info(f"Example candidate: {json.dumps(candidate_data[0], indent=2)[:200]}...")

    # Track positions using conversation format
    all_chunk_ids, input_ids, value_positions, query_position = tracker.track_positions_from_conversation(
        conversation,
        candidate_data
    )

    # Verify that the last integer of query_position is within the range of input_ids
    if query_position:
        assert query_position[-1] < len(input_ids), \
            f"Query position end {query_position[-1]} out of bounds (input_ids length: {len(input_ids)})"


    # Calculate tracking statistics
    imp_indices = []
    imp_indices.extend(value_positions)
    # imp_indices.extend(query_position)

    # imp_indices = list(range(len(input_ids)+1))

    logger.debug(f"Tracked token positions from candidate values: {len(imp_indices)} tokens, {imp_indices}")
    logger.info(f"Number of chunks in all_chunk_ids: {len(all_chunk_ids)}")



    # Initialize cache collection
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    cache_metadata['collect'] = False
    cache_metadata['check'] = False

    # Collect KV cache
    num_layer = 32  # Number of layers in Mistral-7B
    chunk_past_key_values = []

    cache_metadata['collect'] = True

    # Collect KV cache for each block using conversation format
    for i in range(len(all_chunk_ids)):
        if i == 0:
            # Handle system + initial conversation
            initial_conversation = conversation[:3] if len(conversation) >= 3 else conversation
            # Use chat method for consistent formatting
            llm.chat(initial_conversation, sampling_params=SamplingParams(temperature=0.1, max_tokens=1), use_tqdm=False)
        elif i < len(all_chunk_ids) - 1:
            # Handle candidate items pairs (user + assistant)
            candidate_idx = i - 1
            candidate_start_idx = 3 + candidate_idx * 2
            if candidate_start_idx + 1 < len(conversation):
                candidate_pair = conversation[candidate_start_idx:candidate_start_idx + 2]
                # Create a mini conversation with system context for proper formatting
                mini_conversation = [conversation[0]] + candidate_pair
                llm.chat(mini_conversation, sampling_params=SamplingParams(temperature=0.1, max_tokens=1), use_tqdm=False)
        else:
            # Handle final query - use full conversation for final chunk
            llm.chat(conversation, sampling_params=SamplingParams(temperature=0.1, max_tokens=1), use_tqdm=False)

        # Get KV cache from each layer of the model
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers

        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv

            if i == 0:
                # Handle prefix
                temp_k = past_key_values[0].clone()
                temp_v = past_key_values[1].clone()
                # The shape of temp_k/temp_v from hack_kv is (num_tokens_in_prompt_including_bos, ...)
                # Since all_chunk_ids[i] had BOS removed, the KV cache collected corresponds to tokens *with* BOS.
                # We need to be careful with shapes if we intend to concatenate BOS-less KVs.
                # Original logic:
                # if i == 0: temp_k = past_key_values[0].clone() -> includes BOS KV
                # else: temp_k = past_key_values[0][1:].clone() -> removes BOS KV from this chunk
                # This implies the first chunk's KV retains its BOS, subsequent ones have it stripped before cat.

                chunk_past_key_values.append([temp_k, temp_v]) # temp_k/v here includes the BOS position
            else:
                # Handle subsequent content, skip BOS token
                # past_key_values[0] is from a prompt that was tokenized with BOS.
                # So past_key_values[0][0] is KV for BOS, past_key_values[0][1:] is for the actual tokens in all_chunk_ids[i]
                temp_k = past_key_values[0][1:].clone() # (num_heads, seq_len_no_bos, head_dim)
                temp_v = past_key_values[1][1:].clone() # (num_heads, seq_len_no_bos, head_dim)

                # Ensure correct dimension for concatenation if shapes are (num_layers, num_heads, seq_len, head_size)
                # The original shapes from hack_kv are likely (num_tokens, num_heads, head_dim) or similar
                # Let's assume clone and cat dimensions are correct as per original working code.
                # The key is that for chunk i > 0, we skip the first token's KV cache (BOS).
                # The shape from `hack_kv` is (seq_len, num_attn_heads, head_size)

                # actual_kv_k = past_key_values[0][1:] # Skip BOS token's KV for non-first chunks
                # actual_kv_v = past_key_values[1][1:]

                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_v), dim=0)

        logger.debug(f"Processed KV for chunk {i}, num_tokens in chunk (BOS-less): {len(all_chunk_ids[i])}, KV cache collected for {past_key_values[0].shape[0]} tokens (with BOS)")

    # Set merged KV cache
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
    logger.info(f"Final KV cache shape: {chunk_past_key_values[0][0].shape[0]}")


    # Generate using cache
    cache_metadata["check"] = True
    cache_metadata['collect'] = False
    # cache_metadata['suffix_len'] = len(query_ids) # Suffix length
    cache_metadata['imp_indices'] = imp_indices

    print(f"input_ids_len: {len(input_ids)}")
    print(f"imp_indices_len: {len(cache_metadata['imp_indices'])}")
    print(f"First 10 imp_indices: {cache_metadata['imp_indices'][:10] if cache_metadata['imp_indices'] else 'None'}")

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    
    # Use chat method with conversation format
    output = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)

    print(f"Generation with cache: {output[0].outputs[0].text}")
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")
    # return output[0].outputs[0].text # Return output text

    cache_metadata["check"] = False
    cache_metadata['collect'] = False
    output = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")

if __name__ == "__main__":
    user_id = "user_A10FW892S59ABJ"
    logger.info("\n=====Generating Recommendations=====")
    # recommendation = generate_recommendation_with_cache(user_id) # Function doesn't return anymore
    generate_recommendation_with_cache(user_id)
    logger.info("===== End =====")
