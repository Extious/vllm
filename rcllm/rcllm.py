import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer

# Initialize the large model
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95, enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm.set_tokenizer(tokenizer)

class PromptFieldTracker:
    """Track the position information of each candidate JSON field in the prompt"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.item_positions = []  # Store the position information of each candidate
        self.input_ids = None  # Final complete input_ids
    
    def track_positions(self, history_prompt, candidate_data, query_prompt):
        """
        Track the position of each candidate JSON field in the complete prompt
        
        Args:
            history_prompt: Prompt for historical purchase records
            candidate_data: List of candidate data
            query_prompt: Query prompt
        
        Returns:
            Complete input_ids and all_chunk_ids for assembling KV cache
        """
        # Construct the complete prompt using a single concatenation method
        full_prompt = history_prompt
        prefix_ids = self.tokenizer.encode(history_prompt)[1:]  # Remove BOS token
        
        # Create separate prompts for each candidate and construct all_chunk_ids
        candidate_ids = []
        
        for i, item_data in enumerate(candidate_data):
            # Add identifier prefix for each candidate
            item_prefix = f"[{i}]: "
            
            # Ensure we have a string form for index lookup 
            item_json_str = json.dumps(item_data)
            
            candidate_prompt = item_prefix + item_json_str
            candidate_ids.append(self.tokenizer.encode(candidate_prompt)[1:])
            
            # Update the complete prompt
            full_prompt += candidate_prompt + "\n"
            
            # Track the position of each field value
            field_positions = {}
            
            # Calculate the starting position of the current candidate in the complete prompt
            total_prefix_length = len(history_prompt)
            for idx in range(i):
                prev_item_str = json.dumps(candidate_data[idx])
                total_prefix_length += len(f"[{idx}]: {prev_item_str}\n")
            
            item_start_pos = total_prefix_length + len(item_prefix)
            
            # Calculate the position for each field separately
            for field_name, field_value in item_data.items():
                print(f"  Processing field: {field_name} = {str(field_value)[:100]}")
                
                # Special handling for complex data types
                if isinstance(field_value, (dict, list)):
                    # For complex types, we can locate by key name
                    key_with_quotes = f'"{field_name}"'
                    key_pos = item_json_str.find(key_with_quotes)
                    
                    if key_pos != -1:
                        # After finding the key name, extract the entire value
                        colon_pos = item_json_str.find(':', key_pos + len(key_with_quotes))
                        
                        if colon_pos != -1:
                            # Skip whitespace after the colon
                            value_start = colon_pos + 1
                            while value_start < len(item_json_str) and item_json_str[value_start].isspace():
                                value_start += 1
                            
                            # Determine the type and end position of the value
                            if value_start < len(item_json_str):
                                start_char = item_json_str[value_start]
                                
                                # Start marker for nested structures
                                if start_char == '{':  # Dictionary
                                    # Find the matching closing brace
                                    brace_count = 1
                                    value_end = value_start + 1
                                    
                                    while brace_count > 0 and value_end < len(item_json_str):
                                        if item_json_str[value_end] == '{':
                                            brace_count += 1
                                        elif item_json_str[value_end] == '}':
                                            brace_count -= 1
                                        value_end += 1
                                        
                                elif start_char == '[':  # List
                                    # Find the matching closing bracket
                                    bracket_count = 1
                                    value_end = value_start + 1
                                    
                                    while bracket_count > 0 and value_end < len(item_json_str):
                                        if item_json_str[value_end] == '[':
                                            bracket_count += 1
                                        elif item_json_str[value_end] == ']':
                                            bracket_count -= 1
                                        value_end += 1
                                else:
                                    # Unrecognized structure, try to find the next comma or closing brace
                                    value_end = min(
                                        float('inf') if (pos := item_json_str.find(',', value_start)) == -1 else pos,
                                        float('inf') if (pos := item_json_str.find('}', value_start)) == -1 else pos
                                    )
                                    if value_end == float('inf'):
                                        value_end = len(item_json_str)
                                
                                # Extract the value string
                                value_str = item_json_str[value_start:value_end]
                                
                                # Calculate token position
                                field_in_prompt_pos = item_start_pos + value_start
                                field_end_in_prompt = item_start_pos + value_end
                                
                                # Calculate token position
                                prefix_to_field = full_prompt[:field_in_prompt_pos]
                                prefix_to_end = full_prompt[:field_end_in_prompt]
                                
                                value_start_token = len(self.tokenizer.encode(prefix_to_field)) - 1
                                value_end_token = len(self.tokenizer.encode(prefix_to_end)) - 1
                                
                                # Store field position information
                                field_positions[field_name] = {
                                    'value': field_value,  # Store the original value
                                    'start': value_start_token,
                                    'end': value_end_token
                                }
                                
                                print(f"  Complex field {field_name} position: tokens[{value_start_token}:{value_end_token}]")
                                continue
                
                # Standard handling for strings and basic types
                # Construct possible representations based on field value type and content
                possible_formats = []
                
                if isinstance(field_value, str):
                    # String values may have different quote representations
                    clean_value = field_value.replace('"', '\\"').replace("'", "\\'")
                    possible_formats.extend([
                        f'"{field_name}": "{clean_value}"',
                        f'"{field_name}":"{clean_value}"',
                        f'"{field_name}": \'{clean_value}\'',
                        f'"{field_name}":\'{clean_value}\''
                    ])
                    
                    # Handle HTML-escaped apostrophes
                    if "&#39;" in field_value:
                        html_escaped = field_value.replace("&#39;", "'")
                        possible_formats.extend([
                            f'"{field_name}": "{html_escaped}"',
                            f'"{field_name}":"{html_escaped}"'
                        ])
                else:
                    # Non-string values
                    possible_formats.extend([
                        f'"{field_name}": {field_value}',
                        f'"{field_name}":{field_value}'
                    ])
                
                # Find the position of the field in the JSON string
                field_pos = -1
                field_marker = None
                
                # First try exact match
                for fmt in possible_formats:
                    field_pos = item_json_str.find(fmt)
                    if field_pos != -1:
                        field_marker = fmt
                        print(f"    Found exact match: {fmt}")
                        break
                
                # If exact match fails, locate by key name
                if field_pos == -1:
                    print(f"    No exact match found, trying to locate by key name...")
                    key_pattern = f'"{field_name}"'
                    key_pos = item_json_str.find(key_pattern)
                    
                    if key_pos != -1:
                        # After finding the key name, find the colon and value position
                        colon_pos = item_json_str.find(':', key_pos + len(key_pattern))
                        
                        if colon_pos != -1:
                            # Determine the start position of the value
                            value_start = colon_pos + 1
                            while value_start < len(item_json_str) and item_json_str[value_start].isspace():
                                value_start += 1
                            
                            # Determine the end position of the value
                            if value_start < len(item_json_str):
                                if item_json_str[value_start] in ['"', "'"]:
                                    # String value
                                    quote_char = item_json_str[value_start]
                                    value_end = value_start + 1
                                    
                                    # Handle escaped quotes
                                    while value_end < len(item_json_str):
                                        if item_json_str[value_end] == quote_char and item_json_str[value_end-1] != '\\':
                                            value_end += 1  # Include the closing quote
                                            break
                                        value_end += 1
                                    
                                    if value_end >= len(item_json_str):
                                        value_end = len(item_json_str)
                                else:
                                    # Non-string value, find the next comma or closing brace
                                    value_end = min(
                                        float('inf') if (pos := item_json_str.find(',', value_start)) == -1 else pos,
                                        float('inf') if (pos := item_json_str.find('}', value_start)) == -1 else pos
                                    )
                                    if value_end == float('inf'):
                                        value_end = len(item_json_str)
                                
                                # Extract the field marker
                                field_marker = item_json_str[key_pos:value_end]
                                field_pos = key_pos
                                print(f"    Located by key name: {field_marker}")
                
                if field_pos == -1 or field_marker is None:
                    print(f"  Warning: Field {field_name} not found")
                    continue
                
                # Calculate the position of the value in the field
                value_pos = -1
                value_length = 0
                
                if isinstance(field_value, str):
                    # Find the quoted value in the field marker
                    for quote in ['"', "'"]:
                        clean_value = field_value.replace('"', '\\"').replace("'", "\\'")
                        quoted_value = f"{quote}{clean_value}{quote}"
                        value_pos = field_marker.find(quoted_value)
                        
                        if value_pos != -1:
                            value_length = len(quoted_value)
                            break
                    
                    # Handle HTML entities
                    if value_pos == -1 and "&#39;" in field_value:
                        html_escaped = field_value.replace("&#39;", "'")
                        for quote in ['"', "'"]:
                            quoted_value = f"{quote}{html_escaped}{quote}"
                            value_pos = field_marker.find(quoted_value)
                            if value_pos != -1:
                                value_length = len(quoted_value)
                                break
                    
                    # If still not found, try to find the value directly
                    if value_pos == -1:
                        value_pos = field_marker.find(field_value)
                        if value_pos != -1:
                            value_length = len(field_value)
                else:
                    # Non-string value
                    str_value = str(field_value)
                    value_pos = field_marker.find(str_value)
                    if value_pos != -1:
                        value_length = len(str_value)
                
                if value_pos == -1:
                    # If still not found, try to infer the position by the colon
                    colon_pos = field_marker.find(':')
                    if colon_pos != -1:
                        value_pos = colon_pos + 1
                        while value_pos < len(field_marker) and field_marker[value_pos].isspace():
                            value_pos += 1
                        
                        # Find the end position of the value
                        if field_marker[value_pos] in ['"', "'"]:
                            quote_char = field_marker[value_pos]
                            value_end = field_marker.find(quote_char, value_pos + 1)
                            while value_end != -1 and field_marker[value_end - 1] == '\\':
                                value_end = field_marker.find(quote_char, value_end + 1)
                            
                            if value_end != -1:
                                value_length = value_end - value_pos + 1
                            else:
                                value_length = len(field_marker) - value_pos
                        else:
                            # Non-string value, find the next comma or closing brace
                            comma_pos = field_marker.find(',', value_pos)
                            brace_pos = field_marker.find('}', value_pos)
                            
                            if comma_pos != -1 and (brace_pos == -1 or comma_pos < brace_pos):
                                value_length = comma_pos - value_pos
                            elif brace_pos != -1:
                                value_length = brace_pos - value_pos
                            else:
                                value_length = len(field_marker) - value_pos
                
                if value_pos == -1:
                    print(f"  Warning: Unable to locate value position for {field_name}")
                    continue
                
                # Calculate the position of the value in the complete JSON string
                value_in_json_pos = field_pos + value_pos
                
                # Calculate the position of the value in the complete prompt
                value_in_prompt_pos = item_start_pos + value_in_json_pos
                value_end_in_prompt = value_in_prompt_pos + value_length
                
                # Calculate token position
                prefix_to_field = full_prompt[:value_in_prompt_pos]
                prefix_to_end = full_prompt[:value_end_in_prompt]
                
                value_start_token = len(self.tokenizer.encode(prefix_to_field)) - 1
                value_end_token = len(self.tokenizer.encode(prefix_to_end)) - 1
                
                # Validate token boundaries
                if value_start_token >= value_end_token:
                    print(f"  Warning: Invalid token boundaries {value_start_token} >= {value_end_token}")
                    # Try to adjust the boundaries
                    value_end_token = value_start_token + 1
                
                # Store field position information
                field_positions[field_name] = {
                    'value': field_value,
                    'start': value_start_token,
                    'end': value_end_token
                }
                
                print(f"  Field {field_name} position: tokens[{value_start_token}:{value_end_token}]")
            
            # Store the position information of the current candidate
            self.item_positions.append({
                'item_index': i,
                'fields': field_positions
            })
        
        # Add the query prompt
        full_prompt += query_prompt
        query_ids = self.tokenizer.encode(query_prompt)[1:]
        
        # Generate the final input_ids
        self.input_ids = self.tokenizer.encode(full_prompt)[1:]
        
        # Validate whether the field positions are correct
        print("\nVerifying field position accuracy...")
        self.verify_positions()
        
        # Return the complete input_ids and all_chunk_ids for assembling KV cache
        all_chunk_ids = [prefix_ids] + candidate_ids + [query_ids]
        return self.input_ids, all_chunk_ids
    
    def verify_positions(self):
        """Validate whether the extracted positions are accurate"""
        if self.input_ids is None:
            print("Warning: input_ids not generated yet, cannot verify positions")
            return False
            
        all_matched = True
        for item_info in self.item_positions:
            i = item_info['item_index']
            print(f"\nCandidate {i} field positions:")
            
            for field_name, pos in item_info['fields'].items():
                print(f"  {field_name}: [{pos['start']}:{pos['end']}]")
                decoded = self.tokenizer.decode(self.input_ids[pos['start']:pos['end']])
                
                # Format the field value for comparison
                expected_value = pos['value']
                if isinstance(expected_value, (dict, list)):
                    expected_str = json.dumps(expected_value)
                else:
                    expected_str = str(expected_value)
                
                print(f"    Expected value: {expected_str[:100]}")
                print(f"    Decoded value: {decoded[:100]}")
                
                # Check if they match
                if not self._is_similar(expected_value, decoded):
                    print(f"    Warning: Values don't match!")
                    all_matched = False
        
        return all_matched
    
    def visualize_positions(self):
        """Visualize the field marker positions in the entire input text"""
        if self.input_ids is None:
            print("Warning: input_ids not generated yet, cannot visualize positions")
            return
        
        # Decode the complete input
        full_text = self.tokenizer.decode(self.input_ids)
        
        # Create a marked string
        marked_text = list(full_text)
        
        # Use different colors or markers to represent different fields
        markers = {
            'title': '↓T↓',
            'price': '↓P↓',
            'category': '↓C↓',
            'brand': '↓B↓',
            'description': '↓D↓'
        }
        
        # Store all positions and corresponding markers
        positions = []
        
        for item_info in self.item_positions:
            item_idx = item_info['item_index']
            
            for field_name, pos in item_info['fields'].items():
                # Calculate the character position in the actual text (approximate)
                start_text_pos = len(self.tokenizer.decode(self.input_ids[:pos['start']]))
                end_text_pos = len(self.tokenizer.decode(self.input_ids[:pos['end']]))
                
                marker = markers.get(field_name, '↓X↓')
                positions.append((start_text_pos, marker, item_idx, field_name, 'start'))
                positions.append((end_text_pos, marker, item_idx, field_name, 'end'))
        
        # Sort by position and insert markers from back to front (to avoid position shift)
        positions.sort(reverse=True)
        
        for pos, marker, item_idx, field_name, pos_type in positions:
            if 0 <= pos < len(marked_text):
                if pos_type == 'start':
                    marked_text.insert(pos, f"[i{item_idx}:{field_name}>")
                else:
                    marked_text.insert(pos, f"<{field_name}]")
        
        # Print the marked text
        print("\n===== Field Position Visualization =====")
        marked_str = ''.join(marked_text)
        print(marked_str)
        print("===== Visualization End =====\n")
    
    def _is_similar(self, expected, actual):
        """Check if two strings are basically similar (considering possible tokenizer differences)"""
        # Handle numeric types
        if isinstance(expected, (int, float)):
            expected = str(expected)
        
        # Handle dictionary and list types
        if isinstance(expected, (dict, list)):
            expected = json.dumps(expected)
        
        # Handle string types
        expected = str(expected).strip('"\'').strip()
        actual = actual.strip('"\'').strip()
        
        # Direct comparison (considering the case of being exactly the same)
        if expected == actual:
            return True
        
        # Handle special characters, such as apostrophes that may be encoded differently
        expected_clean = expected.replace("'", "'").replace('"', '"').replace("&#39;", "'")
        actual_clean = actual.replace("'", "'").replace('"', '"').replace("&#39;", "'")
        
        if expected_clean == actual_clean:
            return True
        
        # Try to parse JSON strings for comparison
        try:
            expected_json = json.loads(expected.replace("'", '"'))
            actual_json = json.loads(actual.replace("'", '"'))
            if expected_json == actual_json:
                return True
        except:
            pass
        
        # Calculate similarity - remove all non-alphanumeric characters
        expected_normalized = ''.join(c.lower() for c in expected if c.isalnum())
        actual_normalized = ''.join(c.lower() for c in actual if c.isalnum())
        
        if expected_normalized == actual_normalized:
            return True
        
        # Consider partial match cases
        if len(expected_normalized) > 5 and len(actual_normalized) > 5:
            # If one is a substring of the other
            if expected_normalized in actual_normalized or actual_normalized in expected_normalized:
                return True
        
        return False
    
    def get_field_positions(self):
        """Get all field position information"""
        return self.item_positions

def load_user_data(user_id):
    """Load user historical purchase data and candidates"""
    user_dir = os.path.join(os.path.dirname(__file__),"../../dataset", user_id)
    
    # Read historical purchase records
    history_path = os.path.join(user_dir, "history.json")
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    # Read candidates
    candidate_path = os.path.join(user_dir, "candidate.json")
    with open(candidate_path, 'r') as f:
        candidate_data = json.load(f)
        
    return history_data, candidate_data

def generate_recommendation_with_cacheblend(user_id):
    # Load user data
    history_data, candidate_data = load_user_data(user_id)
    
    # Extract username (from user ID)
    username = user_id.split('_')[-1]
    
    # Construct the basic prompt prefix
    prefix = f"""You are an intelligent assistant that can rank items based on the user's preference. The history items and candidate items are listed below. The prefix of history items should be [history] and the prefix of candidate items should be [i]. i is the identifier of the candidate item. Please rank the candidate items based on the user's history. You should strictly obey the following rules: 
    1. All the candidate items should be included and listed using identifiers, in descending order of the user's preference. The most preferred recommendation item should be listed first.
    2. The output format should be [] > [], where each [] is an identifier, e.g., [1] > [2] > [0].
    3. Only respond with the ranking results, do not say any word or explain. Output in the following JSON format: \n{{\"rank\": \"[] > [] .. > []\"}}.
    4. Do not output anything other than the JSON format data.\n"""
    
    # Format the historical data as part of the prompt
    purchase_history = "\n".join([f"- {item}" for item in history_data])
    history_prompt = f"{prefix}\n{purchase_history}\n\n"
    
    # Create separate prompts for each candidate
    candidate_prompts = []
    for i, item in enumerate(candidate_data):
        candidate_prompt = f"[{i}]: {item}"
        candidate_prompts.append(candidate_prompt)
    
    # Create the query prompt
    query_prompt = f"""\n{len(history_data)} history items and {len(candidate_data)} candidate items are listed above. Be careful that the number of the ranking items is {len(candidate_data)}. Please give me the JSON format data of the ranking results, do not output anything other than the JSON format data."""
    
    

    # Create an instance of PromptFieldTracker
    tracker = PromptFieldTracker(tokenizer)
    
    print(f"Number of loaded candidates: {len(candidate_data)}")
    print("Example candidate:", json.dumps(candidate_data[0], indent=2)[:200] + "...")
    
    # Track positions
    input_ids, all_chunk_ids = tracker.track_positions(history_prompt, candidate_data, query_prompt)
    
    # Visualize positions
    tracker.visualize_positions()
    
    # Validate positions and get validation results
    all_matched = tracker.verify_positions()
    
    # Calculate tracking statistics
    position = []
    for item_info in tracker.get_field_positions():
        for field_name, pos in item_info['fields'].items():
            start = pos['start']
            end = pos['end']
            print(f"Field {field_name} position: [{start}:{end}]")
            position.extend(list(range(start, end + 1)))

    # Get token IDs
    prefix_ids = tokenizer.encode(history_prompt)[1:]  # Remove BOS token
    candidate_ids = [tokenizer.encode(prompt)[1:] for prompt in candidate_prompts]
    query_ids = tokenizer.encode(query_prompt)[1:]
    
    print(f"Number of candidates: {len(candidate_ids)}")

    # 获取query部分的token位置
    query_start = len(prefix_ids) + sum(len(ids) for ids in candidate_ids)
    query_end = query_start + len(query_ids) - 1
    print(f"Query token position: [{query_start}:{query_end}]")
    position.extend(list(range(query_start, query_end + 1)))
    # 确保position中的位置不超过input_ids的边界
    input_ids_len = len(input_ids)
    position = [p for p in position if p < input_ids_len]
    print(f"Position length after boundary check: {len(position)}")
    
    # Initialize cache collection
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    cache_metadata['collect'] = False
    cache_metadata['check'] = False
    
    # Organize input sequences
    all_chunk_ids = [prefix_ids] + candidate_ids + [query_ids]
    
    # Collect KV cache
    num_layer = 32  # Number of layers in Mistral-7B
    chunk_past_key_values = []
    
    cache_metadata['collect'] = True
    
    # Collect KV cache for each chunk
    for i in range(len(all_chunk_ids)):
        if i == 0:
            # Handle prefix
            prompt = tokenizer.decode(all_chunk_ids[i])
        else:
            # Handle candidates and query
            prompt = tokenizer.decode(all_chunk_ids[i])
        
        # Generate sampling parameters (only collect KV, not actually generate)
        llm.generate([prompt], SamplingParams(temperature=0.1, max_tokens=1))
        
        # Get KV cache from the model for each layer
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            
            if i == 0:
                # Handle prefix
                temp_k = past_key_values[0].clone()
                temp_v = past_key_values[1].clone()
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                # Handle subsequent content, skip BOS token
                temp_k = past_key_values[0][1:].clone()
                temp_v = past_key_values[1][1:].clone()
                
                # Concatenate KV cache
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_v), dim=0)
        
        print(f"Processed chunk {i}, shape: {temp_k.shape[0]}")
    
    # Set the merged KV cache
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
    print(f"Final KV cache shape: {chunk_past_key_values[0][0].shape[0]}")
    
    # Construct the complete input
    input_ids = []
    for i in range(len(all_chunk_ids)):
        if i == 0:
            input_ids += all_chunk_ids[i]
        else:
            input_ids += all_chunk_ids[i]
    
    input_prompt = tokenizer.decode(input_ids)
    
    # Generate using the cache
    cache_metadata["check"] = True
    cache_metadata['collect'] = False
    # cache_metadata['suffix_len'] = len(query_ids)
    position = list(range(len(input_ids) + 1))
    # 从global_top_diff_positions.txt读取position
    # position_file = os.path.join(os.path.dirname(__file__), 'attn_diff_vis/global_top_diff_positions_80.txt')
    # with open(position_file, 'r') as f:
    #     position = [int(line.strip()) for line in f if line.strip()]

    cache_metadata['imp_indices'] = position
    cache_metadata['prefix_len'] = len(prefix_ids)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    output = llm.generate([input_prompt], sampling_params)
    
    print(f"Generation with cache: {output[0].outputs[0].text}")
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")
    # return output[0].outputs[0].text

    cache_metadata["check"] = False
    cache_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")

if __name__ == "__main__":
    user_id = "user_A1047EDJ84IMAS"

    # Continue the original recommendation generation process
    print("\n=====Generating Recommendations=====")
    recommendation = generate_recommendation_with_cacheblend(user_id)
    
    # # Save the results to a file
    # results_dir = "results"
    # os.makedirs(results_dir, exist_ok=True)
    # with open(os.path.join(results_dir, f"{user_id}_recommendation.txt"), 'w') as f:
    #     f.write(recommendation)
    # print(f"The result of recommendation has been storen in {results_dir}/{user_id}_recommendation.txt")
    print("===== End =====")