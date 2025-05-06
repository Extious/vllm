import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer

# Initialize the large model
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.5,max_model_len=10000)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm.set_tokenizer(tokenizer)

class PromptFieldTracker:
    """Track the position information of each field value for history and candidate items."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Both lists will store dicts like: {'item_index': i, 'fields': {'field_name': {'value': v, 'start': s, 'end': e}}}
        self.candidate_positions = []
        self.history_positions = []
        self.input_ids = None  # Final complete input_ids

    def _find_value_boundaries_in_json(self, json_str, key_name, search_offset=0):
        """Finds the start and end character index of a value for a given key in a JSON string."""
        try:
            key_str_repr = json.dumps(key_name) + ":"
            key_start_index = json_str.find(key_str_repr, search_offset)
            if key_start_index == -1:
                return None, None, search_offset # Key not found

            value_start_in_json = key_start_index + len(key_str_repr)
            while value_start_in_json < len(json_str) and json_str[value_start_in_json].isspace():
                value_start_in_json += 1

            if value_start_in_json >= len(json_str):
                return None, None, value_start_in_json # Value start not found

            # Determine value end
            value_end_in_json = value_start_in_json
            start_char = json_str[value_start_in_json]

            if start_char == '"': # String
                value_end_in_json += 1
                while value_end_in_json < len(json_str):
                    char = json_str[value_end_in_json]
                    prev_char = json_str[value_end_in_json-1]
                    if char == '"' and prev_char != '\\':
                         # Check for escaped backslash before quote: \\"
                         if prev_char == '\\' and value_end_in_json > value_start_in_json + 1 and json_str[value_end_in_json-2] == '\\':
                              pass # It's an escaped backslash, continue
                         else:
                              value_end_in_json += 1; break # End of string found
                    value_end_in_json += 1
                if value_end_in_json > len(json_str): value_end_in_json = len(json_str) # Boundary check
            elif start_char == '{': # Object
                brace_count = 1; value_end_in_json += 1
                while brace_count > 0 and value_end_in_json < len(json_str):
                    if json_str[value_end_in_json] == '{': brace_count += 1
                    elif json_str[value_end_in_json] == '}': brace_count -= 1
                    value_end_in_json += 1
            elif start_char == '[': # Array
                bracket_count = 1; value_end_in_json += 1
                while bracket_count > 0 and value_end_in_json < len(json_str):
                    if json_str[value_end_in_json] == '[': bracket_count += 1
                    elif json_str[value_end_in_json] == ']': bracket_count -= 1
                    value_end_in_json += 1
            else: # Number, boolean, null
                while value_end_in_json < len(json_str) and json_str[value_end_in_json] not in [',', '}', ']']:
                    value_end_in_json += 1

            return value_start_in_json, value_end_in_json, value_end_in_json # Return start, end, next search offset

        except Exception as e:
            print(f"  Error finding boundaries for key '{key_name}': {e}")
            return None, None, search_offset


    def track_positions(self, prefix_prompt, history_data, candidate_data, query_prompt):
        """
        Track the position of each field value for history and candidate items.
        Prompt structure: prefix + history + candidate + query.
        
        Args:
            prefix_prompt: The initial prefix string.
            history_data: List of historical purchase dictionaries.
            candidate_data: List of candidate data dictionaries.
            query_prompt: Query prompt containing instructions.
        
        Returns:
            Complete input_ids (with BOS) and all_chunk_ids (without BOS).
        """
        # Start full_prompt with the prefix
        full_prompt = prefix_prompt
        prefix_chunk_id = self.tokenizer.encode(prefix_prompt)[1:] # Get prefix chunk ID (no BOS)

        history_chunk_ids = []
        candidate_chunk_ids = []
        self.history_positions = [] # Reset
        self.candidate_positions = [] # Reset

        # --- 1. Process History Items ---
        print("Tracking History Value Positions:")
        for i, item_dict in enumerate(history_data):
            try:
                item_json_str = json.dumps(item_dict)
            except TypeError as e:
                print(f"  Warning: Could not serialize history item {i} to JSON: {e}. Skipping.")
                continue

            # Format: "- {json}\n"
            formatted_item = f"- {item_json_str}\n"
            # The prefix before this item starts is the current full_prompt
            prompt_prefix_before_item = full_prompt
            item_prefix_str = "- " # The prefix added before the JSON string

            field_positions = {}
            json_search_offset = 0 # Offset within item_json_str
            for field_name, field_value in item_dict.items():
                value_start_in_json, value_end_in_json, json_search_offset = self._find_value_boundaries_in_json(
                    item_json_str, field_name, json_search_offset
                )

                if value_start_in_json is None: continue

                # Calculate absolute character positions (relative to start of full_prompt)
                base_char_offset = len(prompt_prefix_before_item) + len(item_prefix_str)
                start_char_abs = base_char_offset + value_start_in_json
                end_char_abs = base_char_offset + value_end_in_json

                # Calculate token positions using the full prompt context up to the value start/end
                # The context now includes the initial prefix_prompt
                prefix_to_start_text = prompt_prefix_before_item + formatted_item[:len(item_prefix_str) + value_start_in_json]
                prefix_to_end_text = prompt_prefix_before_item + formatted_item[:len(item_prefix_str) + value_end_in_json]

                start_token = max(0, len(self.tokenizer.encode(prefix_to_start_text)) - 1)
                end_token = max(0, len(self.tokenizer.encode(prefix_to_end_text)) - 1)

                if start_token >= end_token:
                    print(f"  Warning: Invalid token boundaries for history item {i}, field {field_name}: {start_token} >= {end_token}. Adjusting.")
                    end_token = start_token + 1

                field_positions[field_name] = {
                    'value': field_value,
                    'start': start_token,
                    'end': end_token
                }

            self.history_positions.append({
                'item_index': i,
                'fields': field_positions
            })
            # Update the full prompt *after* processing the current item
            full_prompt += formatted_item
            history_chunk_ids.append(self.tokenizer.encode(formatted_item)[1:]) # Add chunk (no BOS)

        # --- 2. Process Candidate Items ---
        print("\nTracking Candidate Value Positions:")
        for i, item_dict in enumerate(candidate_data):
            try:
                item_json_str = json.dumps(item_dict)
            except TypeError as e:
                print(f"  Warning: Could not serialize candidate item {i} to JSON: {e}. Skipping.")
                continue

            # Format: "[{i}]: {json}\n"
            item_prefix_str = f"[{i}]: "
            formatted_item = item_prefix_str + item_json_str + "\n"
            # The prefix before this item starts is the current full_prompt
            prompt_prefix_before_item = full_prompt

            field_positions = {}
            json_search_offset = 0 # Offset within item_json_str
            for field_name, field_value in item_dict.items():
                value_start_in_json, value_end_in_json, json_search_offset = self._find_value_boundaries_in_json(
                    item_json_str, field_name, json_search_offset
                )

                if value_start_in_json is None: continue

                # Calculate absolute character positions
                base_char_offset = len(prompt_prefix_before_item) + len(item_prefix_str)
                start_char_abs = base_char_offset + value_start_in_json
                end_char_abs = base_char_offset + value_end_in_json

                # Calculate token positions
                prefix_to_start_text = prompt_prefix_before_item + formatted_item[:len(item_prefix_str) + value_start_in_json]
                prefix_to_end_text = prompt_prefix_before_item + formatted_item[:len(item_prefix_str) + value_end_in_json]

                start_token = max(0, len(self.tokenizer.encode(prefix_to_start_text)) - 1)
                end_token = max(0, len(self.tokenizer.encode(prefix_to_end_text)) - 1)

                if start_token >= end_token:
                    print(f"  Warning: Invalid token boundaries for candidate item {i}, field {field_name}: {start_token} >= {end_token}. Adjusting.")
                    end_token = start_token + 1

                field_positions[field_name] = {
                    'value': field_value,
                    'start': start_token,
                    'end': end_token
                }

            self.candidate_positions.append({
                'item_index': i,
                'fields': field_positions
            })
            # Update the full prompt *after* processing the current candidate
            full_prompt += formatted_item
            candidate_chunk_ids.append(self.tokenizer.encode(formatted_item)[1:]) # Add chunk (no BOS)

        # --- 3. Add Query Prompt ---
        full_prompt += query_prompt
        query_ids_no_bos = self.tokenizer.encode(query_prompt)[1:]

        # --- 4. Finalize ---
        self.input_ids = self.tokenizer.encode(full_prompt) # Keep BOS

        print("\nVerifying Candidate Value Positions:")
        self.verify_positions(self.candidate_positions, "candidate")
        print("\nVerifying History Value Positions:")
        self.verify_positions(self.history_positions, "history")

        # Combine chunk IDs including the prefix chunk
        all_chunk_ids_no_bos = [prefix_chunk_id] + history_chunk_ids + candidate_chunk_ids + [query_ids_no_bos]
        return self.input_ids, all_chunk_ids_no_bos

    # --- Unified Verification Method ---
    def verify_positions(self, position_data, item_type_name):
        """Validate whether the extracted field value positions are accurate for a given dataset (history or candidate)."""
        if self.input_ids is None:
            print(f"Warning: input_ids not generated yet, cannot verify {item_type_name} positions")
            return False

        all_matched = True
        for item_info in position_data:
            i = item_info['item_index']
            for field_name, pos in item_info['fields'].items():
                start_idx = pos['start'] + 1 # Adjust for BOS
                end_idx = pos['end'] + 1 # Exclusive for slicing

                if start_idx >= end_idx or end_idx > len(self.input_ids):
                     print(f"    Skipping verification for {item_type_name} item {i}, field {field_name} due to invalid range {start_idx}-{end_idx} (len={len(self.input_ids)})")
                     continue

                decoded = self.tokenizer.decode(self.input_ids[start_idx:end_idx])
                expected_value = pos['value'] # Original value

                if not self._is_similar(expected_value, decoded):
                    print(f"    Warning: Values don't match for {item_type_name} item {i}, field {field_name}!")
                    expected_str_repr = repr(expected_value)
                    print(f"      Expected (original): {expected_str_repr[:100]}...")
                    print(f"      Decoded:           {repr(decoded)[:100]}...")
                    all_matched = False

        if all_matched:
            print(f"    All {item_type_name} value positions verified successfully.")
        return all_matched

    # --- Unified Visualization Method ---
    def visualize_positions(self):
        """Visualize the history and candidate value positions in the entire input text"""
        if self.input_ids is None:
            print("Warning: input_ids not generated yet, cannot visualize positions")
            return

        full_text = self.tokenizer.decode(self.input_ids)
        marked_text = list(full_text)
        positions = [] # Store tuples: (char_pos, tag, type)

        # Process history positions
        for item_info in self.history_positions:
            item_idx = item_info['item_index']
            for field_name, pos in item_info['fields'].items():
                start_token_abs = pos['start'] + 1
                end_token_abs = pos['end'] + 1
                if start_token_abs >= end_token_abs or end_token_abs > len(self.input_ids): continue
                try:
                    start_char_pos = len(self.tokenizer.decode(self.input_ids[:start_token_abs]))
                    end_char_pos = len(self.tokenizer.decode(self.input_ids[:end_token_abs]))
                    tag_start = f"[H{item_idx}:{field_name[:1]}>"
                    tag_end = f"<H{item_idx}:{field_name[:1]}]"
                    positions.append((start_char_pos, tag_start, 'start'))
                    positions.append((end_char_pos, tag_end, 'end'))
                except Exception as e:
                    print(f"Error calculating char pos for history {item_idx}:{field_name}: {e}")

        # Process candidate positions
        for item_info in self.candidate_positions:
            item_idx = item_info['item_index']
            for field_name, pos in item_info['fields'].items():
                start_token_abs = pos['start'] + 1
                end_token_abs = pos['end'] + 1
                if start_token_abs >= end_token_abs or end_token_abs > len(self.input_ids): continue
                try:
                    start_char_pos = len(self.tokenizer.decode(self.input_ids[:start_token_abs]))
                    end_char_pos = len(self.tokenizer.decode(self.input_ids[:end_token_abs]))
                    tag_start = f"[C{item_idx}:{field_name[:1]}>"
                    tag_end = f"<C{item_idx}:{field_name[:1]}]"
                    positions.append((start_char_pos, tag_start, 'start'))
                    positions.append((end_char_pos, tag_end, 'end'))
                except Exception as e:
                     print(f"Error calculating char pos for candidate {item_idx}:{field_name}: {e}")

        # Sort by character position, descending, to insert markers correctly
        positions.sort(key=lambda x: x[0], reverse=True)

        for char_pos, tag, _ in positions:
            # Ensure position is valid before inserting
            if 0 <= char_pos <= len(marked_text):
                 marked_text.insert(char_pos, tag)
            else:
                 clamped_pos = max(0, min(char_pos, len(marked_text)))
                 marked_text.insert(clamped_pos, tag + "[?] ") # Mark potentially misplaced tags
                 print(f"Warning: Position {char_pos} out of bounds for marker insertion '{tag}'. Clamped to {clamped_pos}. Max len: {len(marked_text)}")

        marked_str = ''.join(marked_text)
        print("\nVisualized Value Positions:")
        print(marked_str)

    # _is_similar method needs refinement for robustness
    def _is_similar(self, expected, actual):
        """Check if two values are similar, handling type differences and tokenization artifacts."""

        # 1. Basic Cleanup for 'actual' (decoded string)
        actual_str = str(actual).strip() # Remove leading/trailing whitespace

        # 2. Prepare 'expected' based on its type
        if isinstance(expected, (int, float)):
            expected_str = str(expected)
        elif isinstance(expected, bool):
            expected_str = str(expected).lower() # JSON bools are lowercase
        elif isinstance(expected, (dict, list)):
            # For complex types, compare normalized JSON strings
            try:
                expected_json_str = json.dumps(expected, sort_keys=True, separators=(',', ':')) # Compact + sorted
                # Try parsing 'actual' as JSON
                try:
                    actual_json = json.loads(actual_str)
                    actual_json_str = json.dumps(actual_json, sort_keys=True, separators=(',', ':'))
                    if expected_json_str == actual_json_str:
                        return True
                except json.JSONDecodeError:
                    pass # 'actual' wasn't valid JSON, proceed to string comparison
                # Fallback: use non-sorted compact representation for expected
                expected_str = json.dumps(expected, separators=(',', ':'))
            except TypeError:
                expected_str = str(expected) # Fallback if expected can't be dumped
        else: # Assume string type for expected
            expected_str = str(expected)

        # 3. Direct String Comparison (after basic prep)
        if expected_str == actual_str:
            return True

        # 4. String-Specific Normalization and Comparison
        # Only apply these if both are likely strings or can be treated as such
        if isinstance(expected_str, str) and isinstance(actual_str, str):

            # 4a. Normalize whitespace (replace newlines/tabs with spaces, collapse multiple spaces)
            norm_expected = ' '.join(expected_str.split())
            norm_actual = ' '.join(actual_str.split())
            if norm_expected == norm_actual:
                # print("      Info: Matched via normalized whitespace.")
                return True

            # 4b. Compare after stripping outer quotes (handles tokenizer adding/removing quotes)
            # Apply stripping only if normalization didn't match
            strip_expected = norm_expected.strip('\"\'')
            strip_actual = norm_actual.strip('\"\'')
            if strip_expected == strip_actual:
                # print("      Info: Matched via normalized whitespace + stripped quotes.")
                return True

            # 4c. Handle HTML entities like &#39;
            clean_expected = strip_expected.replace("&#39;", "'")
            clean_actual = strip_actual.replace("&#39;", "'")
            if clean_expected == clean_actual:
                 # print("      Info: Matched via normalized whitespace + stripped quotes + clean apos.")
                 return True

            # 4d. Very loose check: compare alphanumeric characters only (case-insensitive)
            # Use this as a last resort as it can be too permissive
            alnum_expected = ''.join(c.lower() for c in expected_str if c.isalnum())
            alnum_actual = ''.join(c.lower() for c in actual_str if c.isalnum())
            if alnum_expected and alnum_actual and alnum_expected == alnum_actual:
                # print(f"      Info: Matched via alnum only: '{alnum_expected}' vs '{alnum_actual}'")
                return True

        # 5. If none of the above matched
        return False


    def get_candidate_positions(self):
        """Get all candidate field value position information"""
        return self.candidate_positions

    def get_history_positions(self):
        """Get all history item field value position information"""
        return self.history_positions

def load_user_data(user_id):
    """Load user historical purchase data and candidates"""
    user_dir = os.path.join(os.path.dirname(__file__),"..", "dataset", user_id)

    # Read historical purchase records (as list of dicts)
    history_path = os.path.join(user_dir, "history.json")
    with open(history_path, 'r') as f:
        history_data = json.load(f)

    # Read candidates (as list of dicts)
    candidate_path = os.path.join(user_dir, "candidate.json")
    with open(candidate_path, 'r') as f:
        candidate_data = json.load(f)

    return history_data, candidate_data


def generate_recommendation_with_cacheblend(user_id):
    # Load user data
    history_data, candidate_data = load_user_data(user_id)

    # Extract username (optional, can be used in prefix)
    username = user_id.split('_')[-1]

    # Define the prefix string
# Note: The history list itself will be added by track_positions based on history_data
    prefix_prompt = f"You are an intelligent assistant that can rank items based on the user's preference.\nAnalyze the provided purchase history and candidate items to identify user preferences and purchase patterns. Then, rank the candidate items based on their alignment with the user's preferences and other contextual factors. All the items should be included and listed using identifiers, in descending order of the user's preference.\n" # Include newline at the end

    # Define the query prompt (instructions only)
    # This assumes history and candidates are already presented before this query.
    query_prompt = f"""\n\n All items related are above. The first {len(history_data)} items are the history items {username} has purchased. The rest {len(candidate_data)} items are candidates.\n
    Please rank the candidates. The most preferred recommendation item should be listed first. The output format should be [] > [], where each [] is an identifier, e.g., [1] > [2]. Only respond with the {len(candidate_data)} JSON format ranking results, do not say any word or explain. Output in the following JSON format:
{{\"rank\": \"[] > [] .. > []\"}}"""


    # Create an instance of PromptFieldTracker
    tracker = PromptFieldTracker(tokenizer)

    # Track positions using the structure: prefix + history + candidate + query
    input_ids_with_bos, all_chunk_ids_no_bos = tracker.track_positions(
        prefix_prompt, history_data, candidate_data, query_prompt
    )

    # Visualize positions (optional)
    # tracker.visualize_positions()

    # Calculate 'imp_indices' from field *values* in both history and candidates
    position = []
    print("\nCalculating Important Indices (from Field Values):")
    # Add history field value positions
    print("  Adding history field value positions...")
    for item_info in tracker.get_history_positions():
        for field_name, pos in item_info['fields'].items():
            start = pos['start']
            end = pos['end']
            if start < end:
                 position.extend(list(range(start, end)))
            else:
                 print(f"    Skipping invalid history value range: item={item_info['item_index']}, field={field_name}, start={start}, end={end}")

    # Add candidate field value positions
    print("  Adding candidate field value positions...")
    for item_info in tracker.get_candidate_positions():
        for field_name, pos in item_info['fields'].items():
            start = pos['start']
            end = pos['end']
            if start < end:
                 position.extend(list(range(start, end)))
            else:
                 print(f"    Skipping invalid candidate value range: item={item_info['item_index']}, field={field_name}, start={start}, end={end}")

    # Remove duplicates and sort
    imp_indices = sorted(list(set(position)))
    print(f"  Total unique important field value indices: {len(imp_indices)}")

    # --- KV Cache Handling ---
    # The full prompt string for generation
    input_prompt = tokenizer.decode(input_ids_with_bos)

    try:
        cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    except AttributeError:
        print("Error: Could not access cache_metadata. CacheBlend features might not work.")
        cache_metadata = {} # Dummy dict to prevent errors if not found

    # Ensure cache_metadata flags are appropriately set before any generation.
    # No explicit "collect" phase (where cache_metadata['collect'] = True for iterative cache building) is performed here.
    
    # Initialize/reset flags before use
    cache_metadata['collect'] = False
    cache_metadata['check'] = False
    cache_metadata['imp_indices'] = None

    # --- Option 1: Generate without Cache (Baseline) / Normal Prefill ---
    print("\n--- Generating without Cache (Normal Prefill) ---")
    # Set flags for this generation call: no collection, no checking (standard prefill)
    cache_metadata["check"] = False
    cache_metadata['collect'] = False # Ensure no collection
    cache_metadata['imp_indices'] = None # No specific indices for this mode

    sampling_params_no_cache = SamplingParams(temperature=0.5, max_tokens=256)
    output_no_cache = llm.generate([input_prompt], sampling_params_no_cache)
    print(f"Normal generation: {output_no_cache[0].outputs[0].text}")
    if hasattr(output_no_cache[0], 'metrics') and output_no_cache[0].metrics:
         try:
             ttft = output_no_cache[0].metrics.first_token_time - output_no_cache[0].metrics.first_scheduled_time
             print(f"TTFT with full prefill: {ttft}")
         except AttributeError:
             print("Metrics structure unexpected, cannot calculate TTFT for normal generation.")
    else:
         print("Metrics not available for normal generation.")
    print("------------")


    # --- Option 2: Generate WITH CacheBlend-like behavior (using imp_indices) ---
    # This relies on vLLM's internal handling of 'imp_indices' if 'check' is True,
    # without an explicit cache collection loop in this script.
    print("\n--- Generating with CacheBlend-like behavior (Using Important Value Indices) ---")
    # Set flags for this generation call: no collection, but check/use imp_indices
    
    cache_metadata["check"] = True
    cache_metadata['collect'] = False # Ensure no collection
    cache_metadata['imp_indices'] = imp_indices # Provide important indices

    sampling_params_cache = SamplingParams(temperature=0.5, max_tokens=256)
    output_cache = llm.generate([input_prompt], sampling_params_cache)

    print(f"Generation with imp_indices: {output_cache[0].outputs[0].text}")
    if hasattr(output_cache[0], 'metrics') and output_cache[0].metrics:
        try:
            ttft_cache = output_cache[0].metrics.first_token_time - output_cache[0].metrics.first_scheduled_time
            print(f"TTFT with imp_indices: {ttft_cache}")
        except AttributeError:
            print("Metrics structure unexpected, cannot calculate TTFT for imp_indices generation.")
    else:
        print("Metrics not available for imp_indices generation.")
    print("------------")

    # Reset cache flags after use to a default state
    cache_metadata["check"] = False
    cache_metadata['collect'] = False
    cache_metadata['imp_indices'] = None

    # Return one of the results, e.g., the one using imp_indices
    return output_cache[0].outputs[0].text


if __name__ == "__main__":
    user_id = "user_A1A0PPF8HE508X"

    # Continue the original recommendation generation process
    print("\n=====Generating Recommendations=====")
    recommendation = generate_recommendation_with_cacheblend(user_id)

    # # Save the results to a file (optional)
    # results_dir = "results"
    # os.makedirs(results_dir, exist_ok=True)
    # output_file = os.path.join(results_dir, f"{user_id}_recommendation.txt")
    # with open(output_file, 'w') as f:
    #     f.write(recommendation if recommendation else "No recommendation generated.")
    # print(f"The result of recommendation has been stored in {output_file}")
    print("===== End =====")
