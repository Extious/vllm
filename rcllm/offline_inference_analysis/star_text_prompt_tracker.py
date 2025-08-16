import json
import logging
import os
from datetime import datetime

# --- Logging configuration ---
# Check if DEBUG_MODE is set (should be imported from main module)
DEBUG_MODE = getattr(__import__('__main__'), 'DEBUG_MODE', True)

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
if not hasattr(logger, '_handlers_configured'):
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
        log_file = os.path.join(log_dir, f"star_text_prompt_tracker_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Tracker log file created: {log_file}")
    
    logger._handlers_configured = True

def get_copurchase_counts(copurchase_df, candidate_item, history_items):
    """Compute co-purchase counts between a candidate and each history item.

    Looks up counts in the provided DataFrame for both directions (item1->item2
    and item2->item1) and sums them. The returned keys follow the convention
    used elsewhere in the project.
    """
    copurchase_info = {}

    candidate_item_id = candidate_item.get('itemID', '')

    for i, history_item in enumerate(history_items):
        history_item_id = history_item.get('itemID', '')

        count1 = copurchase_df[
            (copurchase_df['item1'] == candidate_item_id)
            & (copurchase_df['item2'] == history_item_id)
        ]['copurchase_count'].sum()

        count2 = copurchase_df[
            (copurchase_df['item1'] == history_item_id)
            & (copurchase_df['item2'] == candidate_item_id)
        ]['copurchase_count'].sum()

        total_count = int(count1 + count2)
        copurchase_info[f"Number of users who bought both this item and Item ID {i+1}"] = total_count

    return copurchase_info

class TextPromptTracker:
    """Track position information for text completion format prompts"""
    
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
            logger.debug(f"Complex field '{field_name}' positioning failed, falling back to general search")

        # For simple types (string, number, boolean, null, NaN)
        value_str_repr_for_search = json.dumps(field_value, ensure_ascii=False)
            
        # Strategy 1: Find `value_str_repr_for_search` after `"{field_name}":`
        field_key_marker = f'"{field_name}"'
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

        logger.warning(f"Failed to find char span for field '{field_name}' (value: '{str(field_value)[:50]}{'...' if len(str(field_value)) > 50 else ''}') in JSON")
        logger.debug(f"JSON context: {item_json_str[:200]}{'...' if len(item_json_str) > 200 else ''}")
        return None

    def create_text_completion_prompt(self, user_id: str, history_data: list, candidate_data: list, copurchase_df):
        """
        Create a text completion format prompt from conversation data.
        
        Args:
            user_id: User identifier
            history_data: List of user historical data
            candidate_data: List of candidate item data
            copurchase_df: Copurchase data DataFrame
            
        Returns:
            str: Complete text prompt for text completion
        """
        # Start with system introduction
        prompt_parts = []
        prompt_parts.append("You are an intelligent assistant that can rank items based on the user's preference.")
        prompt_parts.append("")
        
        # Add user history
        prompt_parts.append(f"User {user_id} has purchased the following items in this order:")
        for i, item in enumerate(history_data):
            prompt_parts.append(json.dumps(item, indent=2))
            prompt_parts.append("")
        
        # Add task description
        candidate_count = len(candidate_data)
        prompt_parts.append(f"I will provide you with {candidate_count} items, each indicated by number identifier []. Analyze the user's purchase history to identify preferences and purchase patterns. Then, rank the candidate items based on their alignment with the user's preferences and other contextual factors.")
        prompt_parts.append("")
        
        # Add candidate items with copurchase information
        
        for i, item in enumerate(candidate_data):
            # Get copurchase counts for this candidate item with each history item
            copurchase_info = get_copurchase_counts(copurchase_df, item, history_data)
            
            # Merge copurchase information into the item JSON
            item_with_copurchase = item.copy()
            item_with_copurchase.update(copurchase_info)
            
            # Add candidate item
            prompt_parts.append(f"[{i}]")
            prompt_parts.append(json.dumps(item_with_copurchase, indent=2))
            prompt_parts.append("")
        
        # Add final instructions
        prompt_parts.append(f"""Analyze the user's purchase history to identify user preferences and purchase patterns.
Then, rank the {candidate_count} items above based on their alignment with the user's preferences and other contextual factors.
All the items should be included and listed using identifiers, in descending order of the user's preference.
The most preferred recommendation item should be listed first.
The output format should be [] > [], where each [] is an identifier, e.g., [0] > [1].
Only respond with the ranking results, do not say any word or explain.
Output in the following JSON format: {{ "rank": "[] > [].. > []" }}""")
        
        return "\n".join(prompt_parts)

    def track_positions_from_text(self, full_prompt: str, candidate_data: list):
        """
        Track positions for text completion format.
        
        Args:
            full_prompt: Complete text prompt string
            candidate_data: List of candidate item data (each item is a dictionary)
            
        Returns:
            tuple: (all_chunk_ids, input_ids, value_positions, query_position)
        """
        # Tokenize the full prompt
        prompt_tokens = self.tokenizer.encode(full_prompt, add_special_tokens=True)
        
        # Remove BOS token for consistency with other parts of the system
        if prompt_tokens[0] == self.tokenizer.bos_token_id:
            input_ids = prompt_tokens[1:]
        else:
            input_ids = prompt_tokens
            
        # For chunking, we need to identify different parts of the prompt
        # Let's break it down into logical chunks
        
        # Find the boundaries of different sections
        lines = full_prompt.split('\n')
        
        # Find where candidate items start
        candidate_start_line = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('[0]'):
                candidate_start_line = i
                break
        
        # Find where final instructions start
        final_instructions_start = -1
        for i, line in enumerate(lines):
            if 'Analyze the user\'s purchase history' in line:
                final_instructions_start = i
                break
        
        all_chunk_ids = []
        
        if candidate_start_line != -1:
            # Chunk 1: System intro + history + task description
            prefix_lines = lines[:candidate_start_line]
            prefix_text = '\n'.join(prefix_lines)
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            all_chunk_ids.append(prefix_tokens)
            
            # Chunk 2-N: Each candidate item
            current_line = candidate_start_line
            for i, item_data in enumerate(candidate_data):
                # Find the start and end of this candidate item
                candidate_lines = []
                if current_line < len(lines) and lines[current_line].strip() == f'[{i}]':
                    candidate_lines.append(lines[current_line])  # [i] line
                    current_line += 1
                    
                    # Add JSON lines until we hit the next candidate or final instructions
                    while current_line < len(lines):
                        line = lines[current_line]
                        if line.strip().startswith(f'[{i+1}]') or (final_instructions_start != -1 and current_line >= final_instructions_start):
                            break
                        candidate_lines.append(line)
                        current_line += 1
                
                if candidate_lines:
                    candidate_text = '\n'.join(candidate_lines)
                    candidate_tokens = self.tokenizer.encode(candidate_text, add_special_tokens=False)
                    all_chunk_ids.append(candidate_tokens)
            
            # Final chunk: Instructions
            if final_instructions_start != -1:
                final_lines = lines[final_instructions_start:]
                final_text = '\n'.join(final_lines)
                final_tokens = self.tokenizer.encode(final_text, add_special_tokens=False)
                all_chunk_ids.append(final_tokens)
                
                # Query position is the final chunk
                query_start_token = len(input_ids) - len(final_tokens)
                query_end_token = len(input_ids) - 1
                query_position = list(range(query_start_token, query_end_token + 1))
            else:
                query_position = []
        else:
            # Fallback: treat entire prompt as one chunk
            all_chunk_ids.append(input_ids)
            query_position = []
        
        # Calculate value positions
        value_positions = []
        
        # Find candidate items in the prompt and track their value positions
        for i, item_data in enumerate(candidate_data):
            candidate_marker = f'[{i}]'
            marker_pos = full_prompt.find(candidate_marker)
            
            if marker_pos != -1:
                # Find the JSON content after the marker
                json_start = full_prompt.find('\n', marker_pos) + 1
                next_marker = full_prompt.find(f'[{i+1}]', json_start)
                final_instructions = full_prompt.find('Analyze the user\'s purchase history', json_start)
                
                # Determine where this item's JSON ends
                if next_marker != -1:
                    json_end = next_marker
                elif final_instructions != -1:
                    json_end = final_instructions
                else:
                    json_end = len(full_prompt)
                
                item_json_str = full_prompt[json_start:json_end].strip()
                
                # Track value positions for each field
                for field_name, field_value in item_data.items():
                    span = self._get_value_char_span_in_item_json(field_name, field_value, item_json_str)
                    
                    if span:
                        value_start_in_json, value_end_in_json = span
                        value_start_in_full = json_start + value_start_in_json
                        value_end_in_full = json_start + value_end_in_json
                        
                        # Convert to token positions
                        text_before_value = full_prompt[:value_start_in_full]
                        tokens_before_value = self.tokenizer.encode(text_before_value, add_special_tokens=False)
                        token_start = len(tokens_before_value)
                        
                        text_up_to_value_end = full_prompt[:value_end_in_full]
                        tokens_up_to_value_end = self.tokenizer.encode(text_up_to_value_end, add_special_tokens=False)
                        token_end = len(tokens_up_to_value_end) - 1
                        
                        if token_start <= token_end:
                            value_positions.extend(list(range(token_start, token_end + 1)))
                            logger.debug(f"  Field '{field_name}' value position: tokens [{token_start}:{token_end}] for candidate {i}")
        
        logger.info(f"Text completion format tracking - Query position: {query_position}")
        logger.info(f"Total value positions tracked: {len(value_positions)}")
        logger.debug(f"Value positions summary: {sorted(set(value_positions))[:20]}..." if len(value_positions) > 20 else f"Value positions: {sorted(set(value_positions))}")
        
        return all_chunk_ids, input_ids, value_positions, query_position
