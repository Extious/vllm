import json
import logging

# 获取日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PromptFieldTracker:
    """跟踪提示中特定部分的位置信息"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _get_value_char_span_in_item_json(self, field_name: str, field_value: any, item_json_str: str) -> tuple[int, int] | None:
        """
        在item_json_str中查找field_value的字符起止位置。
        返回 (value_char_start_in_json, value_char_end_in_json) 或 None。
        """
        # 尝试处理复杂类型 (dict, list) 的值
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

                        if start_char == '{':  # 字典
                            brace_count = 1
                            idx = value_char_start_in_json + 1
                            while brace_count > 0 and idx < len(item_json_str):
                                if item_json_str[idx] == '{':
                                    brace_count += 1
                                elif item_json_str[idx] == '}':
                                    brace_count -= 1
                                idx += 1
                            value_char_end_in_json = idx
                        elif start_char == '[':  # 列表
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
            # 如果复杂类型定位失败，会落到下面的通用逻辑，但可能不准确，所以最好在这里处理完
            logger.debug(f"Complex field {field_name} could not be precisely spanned by key search, may rely on general search.")

        # 对于简单类型 (string, number, boolean, null, NaN)
        # json.dumps(field_value) 会生成该值在JSON中的标准字符串表示形式。
        # 例如:
        # - field_value = "example string" -> value_str_repr_for_search = "\"example string\""
        # - field_value = 123            -> value_str_repr_for_search = "123"
        # - field_value = True           -> value_str_repr_for_search = "true"
        # - field_value = None           -> value_str_repr_for_search = "null"
        # - field_value = float('nan')   -> value_str_repr_for_search = "NaN"
        value_str_repr_for_search = json.dumps(field_value, ensure_ascii=False)
            
        # We are looking for `value_str_repr_for_search` directly, or as part of `f'"{field_name}": {value_str_repr_for_search}'`
        
        # Strategy 1: Find `value_str_repr_for_search` after `"{field_name}":`
        field_key_marker = f'"{field_name}"'
        # 在 item_json_str 中查找所有 field_key_marker 出现的位置
        # 使用列表推导式遍历字符串的每个位置，检查从该位置开始是否匹配 field_key_marker
        # 例如，对于 field_key_marker = '"name"'，会在 JSON 字符串中查找所有 '"name"' 的起始位置
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
        跟踪提示中各部分的位置。

        Args:
            prefix_prompt: 提示的前缀部分。
            history_data: 用户历史数据列表。
            candidate_data: 候选项目数据列表 (每个项目是一个字典)。
            query_prompt: 提示的查询/后缀部分。

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

        # --- 收集所有候选块字符串和 tokens ---
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
            
            # 此段包含用于 full_prompt 中下一个元素的分隔符
            full_prompt_str += candidate_str

        # --- 处理 query_prompt ---
        query_tokens = self.tokenizer.encode(query_prompt)[1:]
        # all_chunk_ids = [prefix_and_history_tokens, candidate_tokens, query_tokens]
        all_chunk_ids.append(query_tokens)
        # full_prompt_str = prefix_and_history_str + candidate_str + query_prompt
        full_prompt_str += query_prompt

        # --- 最终的完整提示字符串和 input_ids ---
        # 移除整个序列的 BOS token 来生成 input_ids
        input_ids = []
        for chunk in all_chunk_ids:
            input_ids.extend(chunk)

        # --- 计算 query_position (在 input_ids 中的 token 索引) ---
        # query_tokens 是 query_prompt 的 tokens (已移除 BOS)
        query_start_token = len(input_ids) - len(query_tokens)
        query_end_token = len(input_ids) - 1
        query_position = list(range(query_start_token, query_end_token + 1))
        logger.info(f"Query prompt position: [{query_start_token}:{query_end_token}]")

        # --- 计算 value_positions (候选者值在 input_ids 中的 token 索引) ---
        value_positions = []
        # candidate_item_start_in_full_prompt 指向当前 *候选者块文本的起始处* (例如 "[0]: {...}")
        candidate_item_start_in_full_prompt = len(prefix_and_history_str)

        for i, item_data in enumerate(candidate_data):
            candidate_str = all_candidate_str[i]
            candidate_prefix_str = f"[{i}]: "
            
            # candidate_str 在 full_prompt_str 中的字符起始位置 (不包括 candidate_prefix_str)
            candidate_start_in_full = candidate_item_start_in_full_prompt + len(candidate_prefix_str)

            for field_name, field_value in item_data.items():
                span = self._get_value_char_span_in_item_json(field_name, field_value, candidate_str)

                if span:
                    value_start_in_candidate_str, value_end_in_candidate_str = span # 相对于 candidate_str 起始

                    # 转换为在 full_prompt_str 中的绝对字符位置
                    value_start_in_full_prompt = candidate_start_in_full + value_start_in_candidate_str
                    value_end_in_full_prompt = candidate_start_in_full + value_end_in_candidate_str

                    # 将绝对字符位置转换为 input_ids 中的 token 位置
                    # 使用 add_special_tokens=False 进行长度计算，因为 input_ids 本身是 BOS-less 的单一序列
                    
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
            
            # 将 candidate_item_start_in_full_prompt 移至下一个候选者块的起始处
            candidate_item_start_in_full_prompt += len(candidate_str)
        
        # Compare total tokens and all_chunk_ids
        # 计算并比较两种token序列
        total_tokens = sum(len(chunk) for chunk in all_chunk_ids)
        concatenated_chunks = input_ids
            
        print(f"Token数量比较:")
        print(f"- all_chunk_ids拼接后的token数: {len(concatenated_chunks)}")
        print(f"- input_ids的token数: {len(input_ids)}")
        
        # 检查两个序列是否完全相同
        if concatenated_chunks == input_ids:
            print("✓ all_chunk_ids拼接后与input_ids完全相同")
        else:
            print("✗ all_chunk_ids拼接后与input_ids不同")
            # 找出第一个不同的位置
            import os
            
            # 确保result目录存在
            os.makedirs('./result', exist_ok=True)
            
            # 将完整序列写入文件
            with open('./result/comparation.txt', 'w', encoding='utf-8') as f:
                f.write("=== 完整序列对比 ===\n\n")
                f.write("拼接序列 (concatenated_chunks):\n")
                f.write(self.tokenizer.decode(concatenated_chunks))
                f.write("\n\ninput_ids序列:\n")
                f.write(self.tokenizer.decode(input_ids))
                f.write("\n\n=== 差异分析 ===\n\n")
            
            # 查找并记录第一个差异位置
            for i, (c, j) in enumerate(zip(concatenated_chunks, input_ids)):
                if c != j:
                    diff_msg = f"第一个不同位置在索引 {i}:\n"
                    diff_msg += f"- 拼接序列 token {c}: '{self.tokenizer.decode([c])}'\n"
                    diff_msg += f"- input_ids token {j}: '{self.tokenizer.decode([j])}'\n"
                    
                    # 解码拼接序列的剩余内容
                    remaining_concatenated = concatenated_chunks[i:]
                    decoded_concatenated = self.tokenizer.decode(remaining_concatenated)
                    diff_msg += f"拼接序列剩余内容: {decoded_concatenated[:100]}...\n"
                    
                    # 解码input_ids的剩余内容
                    remaining_input_ids = input_ids[i:]
                    decoded_input_ids = self.tokenizer.decode(remaining_input_ids)
                    diff_msg += f"input_ids剩余内容: {decoded_input_ids[:100]}...\n"
                    
                    # 打印到控制台
                    print(diff_msg)
                    
                    # 追加差异信息到文件
                    with open('./result/comparation.txt', 'a', encoding='utf-8') as f:
                        f.write(diff_msg)
                        
                        # 添加详细的token对比信息
                        f.write("\n=== 详细Token对比 ===\n\n")
                        f.write("索引\t拼接序列token\t拼接序列文本\tinput_ids token\tinput_ids文本\n")
                        f.write("-" * 100 + "\n")
                        
                        # 对比差异位置前后的10个token
                        start_idx = max(0, i - 10)
                        end_idx = min(len(concatenated_chunks), i + 10)
                        
                        for idx in range(start_idx, end_idx):
                            concat_token = concatenated_chunks[idx]
                            input_token = input_ids[idx] if idx < len(input_ids) else "N/A"
                            
                            concat_text = self.tokenizer.decode([concat_token])
                            input_text = self.tokenizer.decode([input_token]) if idx < len(input_ids) else "N/A"
                            
                            # 标记差异位置
                            diff_mark = "***" if idx == i else ""
                            
                            f.write(f"{idx}\t{concat_token}\t{concat_text}\t{input_token}\t{input_text}\t{diff_mark}\n")
                    
                    break
            
        return all_chunk_ids, input_ids, value_positions, query_position 