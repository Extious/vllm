import json
import logging

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PromptFieldTracker:
    """Track position information for specific parts of the prompt"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _get_value_char_span_in_item_json(self, field_name: str, field_value: any, item_json_str: str) -> tuple[int, int] | None:
        """
        Find the character start and end positions of field_value in item_json_str.
        Returns (value_char_start_in_json, value_char_end_in_json) or None.
        """
        # Try to handle complex types (dict, list)
        if isinstance(field_value, (dict, list)):
            key_with_quotes = f'"{field_name}"'
            key_pos = item_json_str.find(key_with_quotes)
            if key_pos != -1:
                colon_pos = item_json_str.find(':', key_pos + len(key_with_quotes))
                if colon_pos != -1:
                    value_char_start_in_json = colon_pos + 1
                    while value_char_start_in_json < len(item_json_str) and item_json_str[value_char_start_in_json].isspace():
                        value_char_start_in_json += 1
                    
                    if value_char_start_in_json < len(item_json_str):
                        start_char = item_json_str[value_char_start_in_json]
                        value_char_end_in_json = -1

                        if start_char == '{':  # Dictionary
                            brace_count = 1
                            idx = value_char_start_in_json + 1
                            while brace_count > 0 and idx < len(item_json_str):
                                if item_json_str[idx] == '{':
                                    brace_count += 1
                                elif item_json_str[idx] == '}':
                                    brace_count -= 1
                                idx += 1
                            value_char_end_in_json = idx
                        elif start_char == '[':  # List
                            bracket_count = 1
                            idx = value_char_start_in_json + 1
                            while bracket_count > 0 and idx < len(item_json_str):
                                if item_json_str[idx] == '[':
                                    bracket_count += 1
                                elif item_json_str[idx] == ']':
                                    bracket_count -= 1
                                idx += 1
                            value_char_end_in_json = idx
                        
                        if value_char_end_in_json != -1:
                            return value_char_start_in_json, value_char_end_in_json
            # If complex type positioning fails, it will fall to the general logic below, but may be inaccurate, so it's best to handle it here
            logger.debug(f"Complex field {field_name} could not be precisely spanned by key search, may rely on general search.")

        # For simple types (string, number, boolean, null, NaN)
        # json.dumps(field_value) will generate the standard string representation of that value in JSON.
        # For example:
        # - field_value = "example string" -> value_str_repr_for_search = "\"example string\""
        # - field_value = 123            -> value_str_repr_for_search = "123"
        # - field_value = True           -> value_str_repr_for_search = "true"
        # - field_value = None           -> value_str_repr_for_search = "null"
        # - field_value = float('nan')   -> value_str_repr_for_search = "NaN"
        value_str_repr_for_search = json.dumps(field_value, ensure_ascii=False)
            
        # We are looking for `value_str_repr_for_search` directly, or as part of `f'"{field_name}": {value_str_repr_for_search}'`
        
        # Strategy 1: Find `value_str_repr_for_search` after `"{field_name}":`
        field_key_marker = f'"{field_name}"'
        # Find all positions where field_key_marker appears in item_json_str
        # Use list comprehension to traverse each position in the string, checking if field_key_marker matches from that position
        # For example, for field_key_marker = '"name"', it will find all starting positions of '"name"' in the JSON string
        key_occurrence_indices = [i for i in range(len(item_json_str)) if item_json_str.startswith(field_key_marker, i)]

        for key_pos in key_occurrence_indices:
            # Look for colon after key
            colon_pos = item_json_str.find(':', key_pos + len(field_key_marker))
            if colon_pos == -1:
                continue

            # Look for value_str_repr_for_search after colon, skipping whitespace
            current_pos = colon_pos + 1
            while current_pos < len(item_json_str) and item_json_str[current_pos].isspace():
                current_pos += 1
            
            if item_json_str.startswith(value_str_repr_for_search, current_pos):
                val_char_start_in_json = current_pos
                val_char_end_in_json = current_pos + len(value_str_repr_for_search)
                return val_char_start_in_json, val_char_end_in_json

        # Strategy 2: (Less precise, if above fails) Try to find value_str_repr_for_search if it's unique enough
        # This is risky if the value string appears elsewhere. Only use as a last resort for simple, unambiguous values.
        # For now, we rely on the key-based search.
        # The original code had more complex logic for various quote styles and missing quotes,
        # but json.dumps standardized representation should simplify this.

        logger.warning(f"Could not reliably find char span for field '{field_name}' with value '{str(field_value)[:50]}' in JSON string: {item_json_str[:100]}...")
        return None

    def track_positions(self, prefix_prompt: str, history_data: list, candidate_data: list, query_prompt: str):
        """
        Track positions of various parts in the prompt.

        Args:
            prefix_prompt: The prefix part of the prompt.
            history_data: List of user historical data.
            candidate_data: List of candidate item data (each item is a dictionary).
            query_prompt: The query/suffix part of the prompt.

        Returns:
            tuple: (all_chunk_ids, input_ids, value_positions, query_position)
        """
        # history_str = "{history_item}\n" + "{history_item}\n" +"{history_item}\n"
        history_str = [f"[history] {item}\n" for item in history_data]

        # prefix_and_history_str = prefix_prompt + history_str
        prefix_and_history_str = prefix_prompt + "".join(history_str)

        # prefix_and_history_tokens = the tokens of prefix_and_history_str
        prefix_and_history_tokens = self.tokenizer.encode(prefix_and_history_str)[1:]

        # all_chunk_ids = [prefix_and_history_tokens]
        all_chunk_ids = [prefix_and_history_tokens]

        # full_prompt_str = prefix_and_history_str
        full_prompt_str = prefix_and_history_str

        # --- Collect all candidate block strings and tokens ---
        all_candidate_str = []

        for i, item_data in enumerate(candidate_data):
            candidate_prefix_str = f"[{i}]:"
            item_str = json.dumps(item_data, ensure_ascii=False)
            # candidate_str = "[i]:{item_str}\n"
            candidate_str = candidate_prefix_str + item_str + "\n"
            # all_candidate_str = ["[0]:{candidate_item}\n", "[1]:{candidate_item}\n", "[2]:{candidate_item}\n"]
            all_candidate_str.append(candidate_str)
            
            candidate_tokens = self.tokenizer.encode(candidate_str)[1:]
            # all_chunk_ids = [prefix_and_history_tokens, candidate_tokens]
            all_chunk_ids.append(candidate_tokens)
            
            # This section contains the separator for the next element in full_prompt
            full_prompt_str += candidate_str

        # --- Process query_prompt ---
        query_tokens = self.tokenizer.encode(query_prompt)[1:]
        # all_chunk_ids = [prefix_and_history_tokens, candidate_tokens, query_tokens]
        all_chunk_ids.append(query_tokens)
        # full_prompt_str = prefix_and_history_str + candidate_str + query_prompt
        full_prompt_str += query_prompt

        # --- Final complete prompt string and input_ids ---
        # Remove the BOS token from the entire sequence to generate input_ids
        input_ids = []
        for chunk in all_chunk_ids:
            input_ids.extend(chunk)

        # --- Calculate query_position (token indices in input_ids) ---
        # query_tokens are the tokens of query_prompt (BOS removed)
        query_start_token = len(input_ids) - len(query_tokens)
        query_end_token = len(input_ids) - 1
        query_position = list(range(query_start_token, query_end_token + 1))
        logger.info(f"Query prompt position: [{query_start_token}:{query_end_token}]")

        # --- Calculate value_positions (token indices of candidate values in input_ids) ---
        value_positions = []
        # candidate_item_start_in_full_prompt points to the start of the current *candidate block text* (e.g., "[0]: {...}")
        candidate_item_start_in_full_prompt = len(prefix_and_history_str)

        for i, item_data in enumerate(candidate_data):
            candidate_str = all_candidate_str[i]
            candidate_prefix_str = f"[{i}]: "
            
            # Character start position of candidate_str in full_prompt_str (excluding candidate_prefix_str)
            candidate_start_in_full = candidate_item_start_in_full_prompt + len(candidate_prefix_str)

            for field_name, field_value in item_data.items():
                span = self._get_value_char_span_in_item_json(field_name, field_value, candidate_str)

                if span:
                    value_start_in_candidate_str, value_end_in_candidate_str = span # Relative to candidate_str start

                    # Convert to absolute character positions in full_prompt_str
                    value_start_in_full_prompt = candidate_start_in_full + value_start_in_candidate_str
                    value_end_in_full_prompt = candidate_start_in_full + value_end_in_candidate_str

                    # Convert absolute character positions to token positions in input_ids
                    # Use add_special_tokens=False for length calculation, because input_ids itself is a BOS-less single sequence
                    
                    text_before_value = full_prompt_str[:value_start_in_full_prompt]
                    tokens_before_value = self.tokenizer.encode(text_before_value, add_special_tokens=False)
                    token_start = len(tokens_before_value)

                    text_up_to_value_end = full_prompt_str[:value_end_in_full_prompt]
                    tokens_up_to_value_end = self.tokenizer.encode(text_up_to_value_end, add_special_tokens=False)
                    token_end = len(tokens_up_to_value_end) - 1 # inclusive end token index
                    
                    actual_value_text_in_prompt = full_prompt_str[value_start_in_full_prompt:value_end_in_full_prompt]

                    if token_start <= token_end:
                        value_positions.extend(list(range(token_start, token_end + 1)))
                        logger.debug(f"  Field '{field_name}' value position: tokens [{token_start}:{token_end}], text: '{actual_value_text_in_prompt[:50]}'")
                    else:
                        logger.warning(f"Token calculation issue for field '{field_name}': value '{str(field_value)[:50]}'. Char span [{value_start_in_full_prompt}:{value_end_in_full_prompt}] resulted in token span [{token_start}:{token_end}]. Actual text: '{actual_value_text_in_prompt}'.")
            
            # Move candidate_item_start_in_full_prompt to the start of the next candidate block
            candidate_item_start_in_full_prompt += len(candidate_str)
        
        # Compare total tokens and all_chunk_ids
        # Calculate and compare the two token sequences
        total_tokens = sum(len(chunk) for chunk in all_chunk_ids)
        concatenated_chunks = input_ids
            
        print(f"Token count comparison:")
        print(f"- Token count after concatenating all_chunk_ids: {len(concatenated_chunks)}")
        print(f"- Token count of input_ids: {len(input_ids)}")

        # Check if the two sequences are completely identical
        if concatenated_chunks == input_ids:
            print("YES: all_chunk_ids concatenated is identical to input_ids")
        else:
            print("NO: all_chunk_ids concatenated differs from input_ids")
            # Find the first different position
            import os

            # Ensure result directory exists
            os.makedirs('./result', exist_ok=True)

            # Write complete sequences to file
            with open('./result/comparation.txt', 'w', encoding='utf-8') as f:
                f.write("=== Complete Sequence Comparison ===\n\n")
                f.write("Concatenated sequence (concatenated_chunks):\n")
                f.write(self.tokenizer.decode(concatenated_chunks))
                f.write("\n\ninput_ids sequence:\n")
                f.write(self.tokenizer.decode(input_ids))
                f.write("\n\n=== Difference Analysis ===\n\n")
            
            # Find and record the first difference position
            for i, (c, j) in enumerate(zip(concatenated_chunks, input_ids)):
                if c != j:
                    diff_msg = f"First different position at index {i}:\n"
                    diff_msg += f"- Concatenated sequence token {c}: '{self.tokenizer.decode([c])}'\n"
                    diff_msg += f"- input_ids token {j}: '{self.tokenizer.decode([j])}'\n"

                    # Decode remaining content of concatenated sequence
                    remaining_concatenated = concatenated_chunks[i:]
                    decoded_concatenated = self.tokenizer.decode(remaining_concatenated)
                    diff_msg += f"Concatenated sequence remaining content: {decoded_concatenated[:100]}...\n"

                    # Decode remaining content of input_ids
                    remaining_input_ids = input_ids[i:]
                    decoded_input_ids = self.tokenizer.decode(remaining_input_ids)
                    diff_msg += f"input_ids remaining content: {decoded_input_ids[:100]}...\n"
                    
                    # Print to console
                    print(diff_msg)

                    # Append difference information to file
                    with open('./result/comparation.txt', 'a', encoding='utf-8') as f:
                        f.write(diff_msg)

                        # Add detailed token comparison information
                        f.write("\n=== Detailed Token Comparison ===\n\n")
                        f.write("Index\tConcat_token\tConcat_text\tinput_ids_token\tinput_ids_text\n")
                        f.write("-" * 100 + "\n")

                        # Compare 10 tokens before and after the difference position
                        start_idx = max(0, i - 10)
                        end_idx = min(len(concatenated_chunks), i + 10)
                        
                        for idx in range(start_idx, end_idx):
                            concat_token = concatenated_chunks[idx]
                            input_token = input_ids[idx] if idx < len(input_ids) else "N/A"
                            
                            concat_text = self.tokenizer.decode([concat_token])
                            input_text = self.tokenizer.decode([input_token]) if idx < len(input_ids) else "N/A"
                            
                            # Mark difference position
                            diff_mark = "***" if idx == i else ""
                            
                            f.write(f"{idx}\t{concat_token}\t{concat_text}\t{input_token}\t{input_text}\t{diff_mark}\n")
                    
                    break
            
        return all_chunk_ids, input_ids, value_positions, query_position

    def track_positions_from_conversation(self, conversation: list, candidate_data: list):
        """
        Track positions for conversation format.
        
        Args:
            conversation: List of conversation messages in chat format
            candidate_data: List of candidate item data (each item is a dictionary)
            
        Returns:
            tuple: (all_chunk_ids, input_ids, value_positions, query_position)
        """
        # Convert conversation to tokens using chat template
        prompt_tokens = self.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False)
        
        # Remove BOS token if present
        if prompt_tokens[0] == self.tokenizer.bos_token_id:
            input_ids = prompt_tokens[1:]
        else:
            input_ids = prompt_tokens
            
        # For chunking, we need to identify different parts of the conversation
        # Let's create chunks based on conversation structure
        all_chunk_ids = []
        current_chunk = []
        
        # Process conversation in chunks
        # First chunk: system + initial user message + assistant ack
        system_and_initial = conversation[:3] if len(conversation) >= 3 else conversation
        initial_tokens = self.tokenizer.apply_chat_template(system_and_initial, tokenize=True, add_generation_prompt=False)
        if initial_tokens[0] == self.tokenizer.bos_token_id:
            initial_tokens = initial_tokens[1:]
        all_chunk_ids.append(initial_tokens)
        
        # Process candidate items in pairs (user + assistant acknowledgment)
        candidate_start_idx = 3
        for i in range(len(candidate_data)):
            if candidate_start_idx + 1 < len(conversation):
                candidate_pair = conversation[candidate_start_idx:candidate_start_idx + 2]
                pair_tokens = self.tokenizer.apply_chat_template(candidate_pair, tokenize=True, add_generation_prompt=False)
                if pair_tokens[0] == self.tokenizer.bos_token_id:
                    pair_tokens = pair_tokens[1:]
                all_chunk_ids.append(pair_tokens)
                candidate_start_idx += 2
        
        # Final query chunk
        if candidate_start_idx < len(conversation):
            final_query = conversation[candidate_start_idx:]
            query_tokens = self.tokenizer.apply_chat_template(final_query, tokenize=True, add_generation_prompt=False)
            if query_tokens[0] == self.tokenizer.bos_token_id:
                query_tokens = query_tokens[1:]
            all_chunk_ids.append(query_tokens)
            
            # Query position is the last chunk
            query_start_token = len(input_ids) - len(query_tokens)
            query_end_token = len(input_ids) - 1
            query_position = list(range(query_start_token, query_end_token + 1))
        else:
            query_position = []
        
        # Calculate value positions
        value_positions = []
        
        # Convert conversation to full text for position tracking
        full_conversation_text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        
        # Find candidate items in the conversation text and track their value positions
        candidate_start_idx = 3
        for i, item_data in enumerate(candidate_data):
            if candidate_start_idx < len(conversation):
                candidate_message = conversation[candidate_start_idx]
                candidate_content = candidate_message["content"]
                
                # Parse the candidate item JSON from the content
                # Format is "[i]\n{json_data}"
                json_start = candidate_content.find('\n') + 1
                if json_start > 0:
                    item_json_str = candidate_content[json_start:]
                    
                    # Find character positions of values in the full conversation text
                    candidate_text_start = full_conversation_text.find(candidate_content)
                    if candidate_text_start != -1:
                        json_start_in_full = candidate_text_start + json_start
                        
                        for field_name, field_value in item_data.items():
                            span = self._get_value_char_span_in_item_json(field_name, field_value, item_json_str)
                            
                            if span:
                                value_start_in_json, value_end_in_json = span
                                value_start_in_full = json_start_in_full + value_start_in_json
                                value_end_in_full = json_start_in_full + value_end_in_json
                                
                                # Convert to token positions
                                text_before_value = full_conversation_text[:value_start_in_full]
                                tokens_before_value = self.tokenizer.encode(text_before_value, add_special_tokens=False)
                                token_start = len(tokens_before_value)
                                
                                text_up_to_value_end = full_conversation_text[:value_end_in_full]
                                tokens_up_to_value_end = self.tokenizer.encode(text_up_to_value_end, add_special_tokens=False)
                                token_end = len(tokens_up_to_value_end) - 1
                                
                                if token_start <= token_end:
                                    value_positions.extend(list(range(token_start, token_end + 1)))
                                    logger.debug(f"  Field '{field_name}' value position: tokens [{token_start}:{token_end}]")
                
                candidate_start_idx += 2
        
        logger.info(f"Conversation format tracking - Query position: {query_position}")
        logger.info(f"Total value positions tracked: {len(value_positions)}")
        
        return all_chunk_ids, input_ids, value_positions, query_position 
