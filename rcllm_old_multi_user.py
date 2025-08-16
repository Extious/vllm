import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import torch
import json
import time
import csv
import glob
from transformers import AutoTokenizer

# 初始化大模型
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.8, enforce_eager=True, max_model_len=10000, dtype="half")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

llm.set_tokenizer(tokenizer)

class PromptFieldTracker:
    """Track position information of each candidate JSON field in the prompt"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.item_positions = []  # 存储每个候选项的位置信息
        self.input_ids = None  # 最终完整的input_ids
        self.query_position = None  # 查询部分的位置信息

    def track_positions(self, history_prompt, candidate_data, query_prompt):
        """
        Track position of each candidate JSON field in complete prompt

        Args:
            history_prompt: History purchase record prompt
            candidate_data: Candidate data list
            query_prompt: Query prompt

        Returns:
            Complete input_ids and all_chunk_ids for KV cache assembly
        """
        # 使用单一连接方法构建完整的提示词
        full_prompt = history_prompt
        prefix_ids = self.tokenizer.encode(history_prompt, add_special_tokens=False)

        # 为每个候选项创建单独的提示词并构建all_chunk_ids
        candidate_ids = []

        for i, item_data in enumerate(candidate_data):
            # 为每个候选项添加标识符前缀
            item_prefix = f"[{i}]: "

            # 确保我们有字符串形式用于索引查找
            item_json_str = json.dumps(item_data, ensure_ascii=False)

            candidate_prompt = item_prefix + item_json_str
            candidate_ids.append(self.tokenizer.encode(candidate_prompt, add_special_tokens=False))

            # 更新完整的提示词
            full_prompt += candidate_prompt + "\n"

            # 追踪每个字段值的位置
            field_positions = {}

            # 计算当前候选项在完整提示词中的起始位置
            total_prefix_length = len(history_prompt)
            for idx in range(i):
                prev_item_str = json.dumps(candidate_data[idx], ensure_ascii=False)
                total_prefix_length += len(f"[{idx}]: {prev_item_str}\n")

            item_start_pos = total_prefix_length + len(item_prefix)

            # 分别计算每个字段的位置
            for field_name, field_value in item_data.items():
                # 查找字段值的精确位置
                value_start_token, value_end_token = self._find_field_value_position(
                    item_json_str, field_name, field_value, item_start_pos, full_prompt
                )
                
                if value_start_token is not None and value_end_token is not None:
                    field_positions[field_name] = {
                        'value': field_value,
                        'start': value_start_token,
                        'end': value_end_token
                    }

            # 存储当前候选项的位置信息
            self.item_positions.append({
                'item_index': i,
                'fields': field_positions
            })

        # 添加查询提示词并追踪其位置
        query_start_pos = len(full_prompt)
        full_prompt += query_prompt
        query_ids = self.tokenizer.encode(query_prompt, add_special_tokens=False)

        # 追踪查询部分的位置
        self._track_query_position(query_start_pos, len(full_prompt), full_prompt)

        # 生成最终的input_ids - 使用一致的编码参数
        self.input_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)

        # 验证字段位置是否正确
        self.verify_positions()

        # 返回完整的input_ids和用于组装KV缓存的all_chunk_ids
        all_chunk_ids = [prefix_ids] + candidate_ids + [query_ids]
        return self.input_ids, all_chunk_ids

    def _find_field_value_position(self, item_json_str, field_name, field_value, item_start_pos, full_prompt):
        """
        精确查找字段值在token序列中的位置
        
        Args:
            item_json_str: JSON字符串
            field_name: 字段名
            field_value: 字段值
            item_start_pos: 当前项在完整提示词中的起始位置
            full_prompt: 完整的提示词
            
        Returns:
            (value_start_token, value_end_token) 或 (None, None)
        """
        try:
            # 查找字段在JSON字符串中的位置
            key_pattern = f'"{field_name}"'
            key_pos = item_json_str.find(key_pattern)
            
            if key_pos == -1:
                return None, None
            
            # 找到冒号位置
            colon_pos = item_json_str.find(':', key_pos + len(key_pattern))
            if colon_pos == -1:
                return None, None
            
            # 跳过冒号后的空白字符
            value_start = colon_pos + 1
            while value_start < len(item_json_str) and item_json_str[value_start].isspace():
                value_start += 1
            
            # 根据值的类型确定结束位置
            if isinstance(field_value, (dict, list)):
                # 复杂类型：查找匹配的括号
                value_end = self._find_complex_value_end(item_json_str, value_start)
            else:
                # 简单类型：查找值的结束位置
                value_end = self._find_simple_value_end(item_json_str, value_start, field_value)
            
            if value_end == -1:
                return None, None
            
            # 计算在完整提示词中的位置
            value_in_prompt_start = item_start_pos + value_start
            value_in_prompt_end = item_start_pos + value_end
            
            # 转换为token位置 - 使用一致的编码参数
            prefix_to_start = full_prompt[:value_in_prompt_start]
            prefix_to_end = full_prompt[:value_in_prompt_end]
            
            value_start_token = len(self.tokenizer.encode(prefix_to_start, add_special_tokens=False))
            value_end_token = len(self.tokenizer.encode(prefix_to_end, add_special_tokens=False))
            
            # 验证token边界的有效性
            if value_start_token >= value_end_token:
                value_end_token = value_start_token + 1
            
            return value_start_token, value_end_token
            
        except Exception as e:
            return None, None

    def _find_complex_value_end(self, json_str, start_pos):
        """查找复杂类型值的结束位置"""
        if start_pos >= len(json_str):
            return -1
            
        start_char = json_str[start_pos]
        
        if start_char == '{':
            # 字典类型
            brace_count = 1
            pos = start_pos + 1
            
            while brace_count > 0 and pos < len(json_str):
                if json_str[pos] == '{':
                    brace_count += 1
                elif json_str[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            return pos if brace_count == 0 else -1
            
        elif start_char == '[':
            # 列表类型
            bracket_count = 1
            pos = start_pos + 1
            
            while bracket_count > 0 and pos < len(json_str):
                if json_str[pos] == '[':
                    bracket_count += 1
                elif json_str[pos] == ']':
                    bracket_count -= 1
                pos += 1
            
            return pos if bracket_count == 0 else -1
        
        return -1

    def _find_simple_value_end(self, json_str, start_pos, expected_value):
        """查找简单类型值的结束位置"""
        if start_pos >= len(json_str):
            return -1
        
        if json_str[start_pos] in ['"', "'"]:
            # 字符串值
            quote_char = json_str[start_pos]
            pos = start_pos + 1
            
            while pos < len(json_str):
                if json_str[pos] == quote_char and (pos == start_pos + 1 or json_str[pos-1] != '\\'):
                    return pos + 1  # 包含结束引号
                pos += 1
            
            return len(json_str)  # 如果没找到结束引号，返回字符串末尾
        else:
            # 数字、布尔值等
            pos = start_pos
            while pos < len(json_str) and json_str[pos] not in [',', '}', ']', '\n']:
                pos += 1
            return pos

    def _track_query_position(self, query_start_pos, query_end_pos, full_prompt):
        """追踪查询部分的位置"""
        try:
            # 计算查询部分的token位置 - 使用一致的编码参数
            prefix_to_start = full_prompt[:query_start_pos]
            prefix_to_end = full_prompt[:query_end_pos]
            
            query_start_token = len(self.tokenizer.encode(prefix_to_start, add_special_tokens=False))
            query_end_token = len(self.tokenizer.encode(prefix_to_end, add_special_tokens=False))
            
            self.query_position = {
                'start': query_start_token,
                'end': query_end_token,
                'char_start': query_start_pos,
                'char_end': query_end_pos
            }
            
        except Exception as e:
            pass

    def verify_positions(self):
        """验证提取的位置是否准确"""
        if self.input_ids is None:
            return False

        all_matched = True
        
        # 验证候选项字段位置
        for item_info in self.item_positions:
            i = item_info['item_index']

            for field_name, pos in item_info['fields'].items():
                if pos['start'] < len(self.input_ids) and pos['end'] <= len(self.input_ids):
                    decoded = self.tokenizer.decode(self.input_ids[pos['start']:pos['end']])
                    
                    # 格式化字段值用于比较
                    expected_value = pos['value']
                    if isinstance(expected_value, (dict, list)):
                        expected_str = json.dumps(expected_value, ensure_ascii=False)
                    else:
                        expected_str = str(expected_value)

                    # 检查是否匹配
                    if not self._is_similar(expected_value, decoded):
                        all_matched = False
                else:
                    all_matched = False
        
        # 验证查询位置
        if self.query_position:
            query_start = self.query_position['start']
            query_end = self.query_position['end']
            
            if query_start < len(self.input_ids) and query_end <= len(self.input_ids):
                decoded_query = self.tokenizer.decode(self.input_ids[query_start:query_end])
            else:
                all_matched = False

        return all_matched

    def visualize_positions(self):
        """可视化字段标记位置在整个输入文本中的分布"""
        if self.input_ids is None:
            return

        # 解码完整输入
        full_text = self.tokenizer.decode(self.input_ids)

        # 创建标记字符串
        marked_text = list(full_text)

        # Use different colors or markers for different fields
        markers = {
            'title': 'Title',
            'price': 'Price',
            'category': 'Category',
            'brand': 'Brand',
            'description': 'Description'
        }

        # 存储所有位置和对应的标记
        positions = []

        for item_info in self.item_positions:
            item_idx = item_info['item_index']

            for field_name, pos in item_info['fields'].items():
                # 计算在实际文本中的字符位置（近似） - 使用一致的解码参数
                start_text_pos = len(self.tokenizer.decode(self.input_ids[:pos['start']], skip_special_tokens=False, clean_up_tokenization_spaces=False))
                end_text_pos = len(self.tokenizer.decode(self.input_ids[:pos['end']], skip_special_tokens=False, clean_up_tokenization_spaces=False))

                marker = markers.get(field_name, 'Unknown')
                positions.append((start_text_pos, marker, item_idx, field_name, 'start'))
                positions.append((end_text_pos, marker, item_idx, field_name, 'end'))

        # 按位置排序并从后往前插入标记（避免位置偏移）
        positions.sort(reverse=True)

        for pos, marker, item_idx, field_name, pos_type in positions:
            if 0 <= pos < len(marked_text):
                if pos_type == 'start':
                    marked_text.insert(pos, f"[i{item_idx}:{field_name}>")
                else:
                    marked_text.insert(pos, f"<{field_name}]")

    def _is_similar(self, expected, actual):
        """检查两个字符串是否基本相似（考虑可能的tokenizer差异）"""
        # 处理数字类型
        if isinstance(expected, (int, float)):
            expected = str(expected)

        # 处理字典和列表类型
        if isinstance(expected, (dict, list)):
            expected = json.dumps(expected, ensure_ascii=False)

        # 处理字符串类型
        expected = str(expected).strip('"\'').strip()
        actual = actual.strip('"\'').strip()

        # 直接比较（考虑完全相同的情况）
        if expected == actual:
            return True

        # 处理特殊字符，如撇号可能编码不同的
        expected_clean = expected.replace("'", "'").replace('"', '"').replace("s Bees", "s Bees")
        actual_clean = actual.replace("'", "'").replace('"', '"').replace("s Bees", "s Bees")

        if expected_clean == actual_clean:
            return True

        # 尝试解析JSON字符串进行比较
        try:
            expected_json = json.loads(expected.replace("'", '"'))
            actual_json = json.loads(actual.replace("'", '"'))
            if expected_json == actual_json:
                return True
        except:
            pass

        # 计算相似度 - 移除所有非字母数字字符
        expected_normalized = ''.join(c.lower() for c in expected if c.isalnum())
        actual_normalized = ''.join(c.lower() for c in actual if c.isalnum())

        if expected_normalized == actual_normalized:
            return True

        # 考虑部分匹配的情况
        if len(expected_normalized) > 5 and len(actual_normalized) > 5:
            # 如果一个是另一个的子字符串
            if expected_normalized in actual_normalized or actual_normalized in expected_normalized:
                return True

        return False

    def get_field_positions(self):
        """Get all field position information"""
        return self.item_positions

    def get_query_position(self):
        """Get query part position information"""
        return self.query_position

def ensure_token_consistency(input_ids, tokenizer):
    """
    Ensure token sequence consistency after decode-encode cycle
    
    Args:
        input_ids: Original token sequence
        tokenizer: Tokenizer instance
        
    Returns:
        Consistent token sequence
    """
    try:
        # 首先尝试标准解码-编码
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        reencoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
        
        # 检查是否保持一致性
        if len(input_ids) == len(reencoded_ids) and all(a == b for a, b in zip(input_ids, reencoded_ids)):
            return reencoded_ids
        
        # 方法2: 尝试分段处理
        chunk_size = 50  # 每次处理50个token
        consistent_chunks = []
        
        for i in range(0, len(input_ids), chunk_size):
            chunk = input_ids[i:i+chunk_size]
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            chunk_reencoded = tokenizer.encode(chunk_text, add_special_tokens=False)
            
            if chunk == chunk_reencoded:
                consistent_chunks.extend(chunk_reencoded)
            else:
                consistent_chunks.extend(chunk)
        
        if len(consistent_chunks) == len(input_ids) and all(a == b for a, b in zip(input_ids, consistent_chunks)):
            return consistent_chunks
        
        # 方法3: 逐token处理
        safe_tokens = []
        accumulated_text = ""
        
        for token_id in input_ids:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            accumulated_text += token_text
            safe_tokens.append(token_id)
        
        # 验证重构的文本
        final_reencoded = tokenizer.encode(accumulated_text, add_special_tokens=False)
        
        if len(input_ids) == len(final_reencoded) and all(a == b for a, b in zip(input_ids, final_reencoded)):
            return final_reencoded
        else:
            return input_ids
                
    except Exception as e:
        return input_ids

def load_user_data(user_id):
    """
    Load user history and candidates from reviewer_data_processed_strict.

    Read single reviewer file, use first 3 as history, rest as candidates.
    Support "user_XXXX" and "reviewer_XXXX" format IDs.
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

    with open(data_file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    if len(all_data) < 4:
        raise ValueError(
            f"Insufficient data for {user_id}: need at least 4 items (3 history + 1 candidate), got {len(all_data)}"
        )

    history_data = all_data[:3]
    candidate_data = all_data[3:]

    return history_data, candidate_data

def test_user_performance(user_id):
    """Test performance data for a single user"""
    try:
        # 加载用户数据
        history_data, candidate_data = load_user_data(user_id)

        # 提取用户名（从用户ID）
        username = user_id.split('_')[-1]

        # 构建基本提示词前缀
        prefix = f"You are an intelligent assistant that can rank items based on the user's preference.\nUser {username} has purchased the following items in this order:"

        # 将历史数据格式化为提示词的一部分
        purchase_history = "\n".join([f"- {item}" for item in history_data])
        history_prompt = f"{prefix}\n{purchase_history}\n\n"

        # 为每个候选项创建单独的提示词
        candidate_prompts = []
        for i, item in enumerate(candidate_data):
            candidate_prompt = f"[{i}]: {item}"
            candidate_prompts.append(candidate_prompt)

        # 创建查询提示词
        query_prompt = f"""Analyze the user's purchase history to identify user preferences and purchase patterns. Then, rank the {len(candidate_prompts)} items above based on their alignment with the user's preferences and other contextual factors. All the items should be included and listed using identifiers, in descending order of the user's preference. The most preferred recommendation item should be listed first. The output format should be [] > [], where each [] is an identifier, e.g., [1] > [2]. Only respond with the ranking results, do not say any word or explain. Output in the following JSON format: \n{{"rank": "[] > [] .. > []"}} Do not output anything other than the json format data."""

        # 创建PromptFieldTracker实例
        tracker = PromptFieldTracker(tokenizer)

        # 追踪位置
        input_ids, all_chunk_ids = tracker.track_positions(history_prompt, candidate_data, query_prompt)

        # 计算指标
        prompt_length = len(input_ids)
        total_item_number = len(history_data) + len(candidate_data)

        # 构建完整的提示词来计算字符长度
        full_prompt = history_prompt
        for i, item in enumerate(candidate_data):
            item_json_str = json.dumps(item, ensure_ascii=False)
            full_prompt += f"[{i}]: {item_json_str}\n"
        full_prompt += query_prompt

        # 获取位置信息
        position = []
        for item_info in tracker.get_field_positions():
            for field_name, pos in item_info['fields'].items():
                start = pos['start']
                end = pos['end']
                position.extend(list(range(start, end + 1)))

        query_position = tracker.get_query_position()
        if query_position:
            query_start = query_position['start']
            query_end = query_position['end']
            position.extend(list(range(query_start, query_end + 1)))

        # 边界检查
        input_ids_length = len(input_ids)
        position = [idx for idx in position if 0 <= idx < input_ids_length]

        # 测试GPU缓存性能
        gpu_results = test_gpu_cache(input_ids, all_chunk_ids, position, full_prompt)
        
        # 测试CPU缓存性能
        cpu_results = test_cpu_cache(input_ids, all_chunk_ids, position, full_prompt)

        return {
            'user_id': user_id,
            'prompt_length': prompt_length,
            'total_item_number': total_item_number,
            'full_prefill_ttft': gpu_results['full_prefill_ttft'],
            'gpu_cache_ttft': gpu_results['gpu_cache_ttft'],
            'cpu_cache_ttft': cpu_results['cpu_cache_ttft'],
            'total_cpu_overhead': cpu_results['total_cpu_overhead']
        }

    except Exception as e:
        print(f"Error processing user {user_id}: {e}")
        return None

def test_gpu_cache(input_ids, all_chunk_ids, position, full_prompt):
    """Test GPU cache performance"""
    # 初始化缓存收集
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    cache_metadata['collect'] = False
    cache_metadata['check'] = False

    # 收集KV缓存
    num_layer = 32
    chunk_past_key_values = []

    cache_metadata['collect'] = True

    # 使用改进方法处理所有chunk
    all_chunk_ids_processed = []
    for i, chunk_ids in enumerate(all_chunk_ids):
        processed_chunk = ensure_token_consistency(chunk_ids, tokenizer)
        all_chunk_ids_processed.append(processed_chunk)

    all_chunk_ids = all_chunk_ids_processed

    # 为每个块收集KV缓存
    for i in range(len(all_chunk_ids)):
        if i == 0:
            prompt = tokenizer.decode(all_chunk_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            prompt = tokenizer.decode(all_chunk_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        # 生成采样参数（只收集KV，不实际生成）
        llm.generate([prompt], SamplingParams(temperature=0.1, max_tokens=1), use_tqdm=False)

        # 从模型获取每层的KV缓存
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers

        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv

            if i == 0:
                temp_k = past_key_values[0].clone()
                temp_v = past_key_values[1].clone()
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                temp_k = past_key_values[0][1:].clone()
                temp_v = past_key_values[1][1:].clone()

                # 连接KV缓存
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_v), dim=0)

    # 设置合并的KV缓存
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values

    # 使用安全的解码方法
    input_prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # 使用缓存生成
    cache_metadata["check"] = True
    cache_metadata['collect'] = False
    cache_metadata['imp_indices'] = position

    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    output = llm.generate([input_prompt], sampling_params, use_tqdm=False)

    gpu_cache_ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time

    # 测试无缓存性能
    cache_metadata["check"] = False
    cache_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params, use_tqdm=False)
    full_prefill_ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time

    return {
        'gpu_cache_ttft': gpu_cache_ttft,
        'full_prefill_ttft': full_prefill_ttft
    }

def test_cpu_cache(input_ids, all_chunk_ids, position, full_prompt):
    """Test CPU cache performance"""
    # 初始化缓存收集
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    cache_metadata['collect'] = False
    cache_metadata['check'] = False

    # 收集KV缓存 - CPU版本
    num_layer = 32
    chunk_past_key_values_cpu = []

    cache_metadata['collect'] = True

    # 使用改进方法处理所有chunk
    all_chunk_ids_processed = []
    for i, chunk_ids in enumerate(all_chunk_ids):
        processed_chunk = ensure_token_consistency(chunk_ids, tokenizer)
        all_chunk_ids_processed.append(processed_chunk)

    all_chunk_ids = all_chunk_ids_processed

    # Collect KV cache for each chunk and store in CPU
    cpu_to_gpu_time = 0.0
    for i in range(len(all_chunk_ids)):
        if i == 0:
            prompt = tokenizer.decode(all_chunk_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            prompt = tokenizer.decode(all_chunk_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        # 生成采样参数（只收集KV，不实际生成）
        llm.generate([prompt], SamplingParams(temperature=0.1, max_tokens=1), use_tqdm=False)

        # 从模型获取每层的KV缓存
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers

        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv

            if i == 0:
                # 处理前缀 - 存储到CPU
                temp_k = past_key_values[0].clone().cpu()
                temp_v = past_key_values[1].clone().cpu()
                chunk_past_key_values_cpu.append([temp_k, temp_v])
            else:
                # 处理后续内容，跳过BOS token - 存储到CPU
                temp_k = past_key_values[0][1:].clone().cpu()
                temp_v = past_key_values[1][1:].clone().cpu()

                # 在CPU上连接KV缓存
                chunk_past_key_values_cpu[j][0] = torch.cat((chunk_past_key_values_cpu[j][0], temp_k), dim=0)
                chunk_past_key_values_cpu[j][1] = torch.cat((chunk_past_key_values_cpu[j][1], temp_v), dim=0)

    # Move CPU cache back to GPU for inference
    chunk_past_key_values_gpu = []
    transfer_start = time.time()
    for j in range(num_layer):
        gpu_k = chunk_past_key_values_cpu[j][0].cuda()
        gpu_v = chunk_past_key_values_cpu[j][1].cuda()
        chunk_past_key_values_gpu.append([gpu_k, gpu_v])
    cpu_to_gpu_time = time.time() - transfer_start

    # Set the merged KV cache
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values_gpu

    # 使用安全的解码方法
    input_prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # 使用缓存生成
    cache_metadata["check"] = True
    cache_metadata['collect'] = False
    cache_metadata['imp_indices'] = position

    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    output = llm.generate([input_prompt], sampling_params, use_tqdm=False)

    cpu_cache_ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    total_cpu_overhead = cpu_to_gpu_time + cpu_cache_ttft

    cache_metadata["check"] = False
    cache_metadata['collect'] = False

    return {
        'cpu_cache_ttft': cpu_cache_ttft,
        'total_cpu_overhead': total_cpu_overhead
    }

def get_user_files(data_dir, max_users=100):
    """Get user file list"""
    pattern = os.path.join(data_dir, "reviewer_*.json")
    files = glob.glob(pattern)
    
    # Limit file count
    if len(files) > max_users:
        files = files[:max_users]
    
    return files

def extract_user_id_from_file(file_path):
    """Extract user ID from file path"""
    filename = os.path.basename(file_path)
    # Remove 'reviewer_' prefix and '.json' suffix
    user_id = filename[len("reviewer_"):-len(".json")]
    return f"reviewer_{user_id}"

def main():
    """Main function: Test multiple users and save results to CSV"""
    print("Starting multi-user performance testing...")
    
    data_dir = "/home/comp/24481750/rcllm/amazon/dataset/reviewer_data_processed_strict"
    output_file = "/home/comp/24481750/rcllm/vllm/multi_user_performance_results_A100.csv"
    
    # Get user file list
    user_files = get_user_files(data_dir, max_users=500)
    print(f"Found {len(user_files)} user files")
    
    # Create CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'user_id', 
            'prompt_length', 
            'total_item_number', 
            'full_prefill_ttft', 
            'gpu_cache_ttft', 
            'cpu_cache_ttft', 
            'total_cpu_overhead'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each user
        for i, file_path in enumerate(user_files):
            user_id = extract_user_id_from_file(file_path)
            print(f"Processing user {i+1}/{len(user_files)}: {user_id}")
            
            try:
                results = test_user_performance(user_id)
                if results:
                    writer.writerow(results)
                    print(f"  Completed - Prompt length: {results['prompt_length']}, Item count: {results['total_item_number']}")
                else:
                    print(f"  Skipped - Processing failed")
            except Exception as e:
                print(f"  Error: {e}")
                continue
            
            # Save every 10 users
            if (i + 1) % 10 == 0:
                csvfile.flush()
                print(f"Processed {i+1} users, intermediate save...")
    
    print(f"Testing completed! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
