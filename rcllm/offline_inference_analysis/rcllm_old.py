import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import torch
import json
import time
from transformers import AutoTokenizer

# 初始化大模型
# llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.8, enforce_eager=True, max_model_len=10000)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

llm.set_tokenizer(tokenizer)

class PromptFieldTracker:
    """追踪每个候选项JSON字段在提示词中的位置信息"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.item_positions = []  # 存储每个候选项的位置信息
        self.input_ids = None  # 最终完整的input_ids
        self.query_position = None  # 查询部分的位置信息

    def track_positions(self, history_prompt, candidate_data, query_prompt):
        """
        追踪每个候选项JSON字段在完整提示词中的位置

        Args:
            history_prompt: 历史购买记录的提示词
            candidate_data: 候选项数据列表
            query_prompt: 查询提示词

        Returns:
            完整的input_ids和用于组装KV缓存的all_chunk_ids
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

            # print(f"\n处理候选项 {i}:")
            # print(f"  JSON字符串: {item_json_str[:200]}...")
            # print(f"  在提示词中的起始位置: {item_start_pos}")
            
            # 分别计算每个字段的位置
            for field_name, field_value in item_data.items():
                # print(f"  正在处理字段: {field_name} = {str(field_value)[:100]}")
                
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
                    # print(f"    字段 {field_name} 位置: tokens[{value_start_token}:{value_end_token}]")
                else:
                    # print(f"    警告: 无法定位字段 {field_name} 的位置")
                    pass

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
        # print("\n验证字段位置精度...")
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
                # print(f"      未找到字段名: {key_pattern}")
                return None, None
            
            # 找到冒号位置
            colon_pos = item_json_str.find(':', key_pos + len(key_pattern))
            if colon_pos == -1:
                # print(f"      未找到冒号")
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
                # print(f"      无法确定值的结束位置")
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
                # print(f"      警告: 无效的token边界 {value_start_token} >= {value_end_token}")
                value_end_token = value_start_token + 1
            
            return value_start_token, value_end_token
            
        except Exception as e:
            # print(f"      错误: 查找字段位置时出现异常: {e}")
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
            
            # print(f"\n查询部分位置:")
            # print(f"  字符位置: [{query_start_pos}:{query_end_pos}]")
            # print(f"  Token位置: [{query_start_token}:{query_end_token}]")
            
        except Exception as e:
            # print(f"追踪查询位置时出现错误: {e}")
            pass

    def verify_positions(self):
        """验证提取的位置是否准确"""
        if self.input_ids is None:
            # print("警告: input_ids尚未生成，无法验证位置")
            return False

        all_matched = True
        
        # 验证候选项字段位置
        for item_info in self.item_positions:
            i = item_info['item_index']
            # print(f"\n候选项 {i} 字段位置:")

            for field_name, pos in item_info['fields'].items():
                # print(f"  {field_name}: [{pos['start']}:{pos['end']}]")
                
                if pos['start'] < len(self.input_ids) and pos['end'] <= len(self.input_ids):
                    decoded = self.tokenizer.decode(self.input_ids[pos['start']:pos['end']])
                    
                    # 格式化字段值用于比较
                    expected_value = pos['value']
                    if isinstance(expected_value, (dict, list)):
                        expected_str = json.dumps(expected_value, ensure_ascii=False)
                    else:
                        expected_str = str(expected_value)

                    # print(f"    期望值: {expected_str[:100]}")
                    # print(f"    解码值: {decoded[:100]}")

                    # 检查是否匹配
                    if not self._is_similar(expected_value, decoded):
                        # print(f"    警告: 值不匹配!")
                        all_matched = False
                    else:
                        # print(f"    匹配成功")
                        pass
                else:
                    # print(f"    错误: token位置超出范围")
                    all_matched = False
        
        # 验证查询位置
        if self.query_position:
            # print(f"\n查询部分验证:")
            query_start = self.query_position['start']
            query_end = self.query_position['end']
            
            if query_start < len(self.input_ids) and query_end <= len(self.input_ids):
                decoded_query = self.tokenizer.decode(self.input_ids[query_start:query_end])
                # print(f"  解码的查询部分: {decoded_query[:200]}...")
                # print(f"  查询位置验证成功")
            else:
                # print(f"  错误: 查询token位置超出范围")
                all_matched = False

        return all_matched

    def visualize_positions(self):
        """可视化字段标记位置在整个输入文本中的分布"""
        if self.input_ids is None:
            # print("警告: input_ids尚未生成，无法可视化位置")
            return

        # 解码完整输入
        full_text = self.tokenizer.decode(self.input_ids)

        # 创建标记字符串
        marked_text = list(full_text)

        # 使用不同的颜色或标记来表示不同的字段
        markers = {
            'title': '标题',
            'price': '价格',
            'category': '类别',
            'brand': '品牌',
            'description': '描述'
        }

        # 存储所有位置和对应的标记
        positions = []

        for item_info in self.item_positions:
            item_idx = item_info['item_index']

            for field_name, pos in item_info['fields'].items():
                # 计算在实际文本中的字符位置（近似） - 使用一致的解码参数
                start_text_pos = len(self.tokenizer.decode(self.input_ids[:pos['start']], skip_special_tokens=False, clean_up_tokenization_spaces=False))
                end_text_pos = len(self.tokenizer.decode(self.input_ids[:pos['end']], skip_special_tokens=False, clean_up_tokenization_spaces=False))

                marker = markers.get(field_name, '未知')
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

        # 打印标记文本
        # print("\n===== 字段位置可视化 =====")
        # marked_str = ''.join(marked_text)
        # print(marked_str)
        # print("===== 可视化结束 =====\n")

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
        """获取所有字段位置信息"""
        return self.item_positions

    def get_query_position(self):
        """获取查询部分的位置信息"""
        return self.query_position

def analyze_token_mismatch(input_ids, tokenizer):
    """深入分析token不匹配的具体原因"""
    # print("=== Token不匹配分析 ===")
    
    # 原始解码
    original_text = tokenizer.decode(input_ids)
    
    # 重新编码
    reencoded_ids = tokenizer.encode(original_text, add_special_tokens=False)
    
    # print(f"原始长度: {len(input_ids)}")
    # print(f"重编码长度: {len(reencoded_ids)}")
    
    # 找到第一个不匹配的位置
    mismatch_start = None
    min_len = min(len(input_ids), len(reencoded_ids))
    
    for i in range(min_len):
        if input_ids[i] != reencoded_ids[i]:
            mismatch_start = i
            break
    
    if mismatch_start is not None:
        # print(f"第一个不匹配位置: {mismatch_start}")
        
        # 分析不匹配前后的context
        context_start = max(0, mismatch_start - 5)
        context_end = min(len(input_ids), mismatch_start + 10)
        
        # print("不匹配前后的token context:")
        # for i in range(context_start, context_end):
        #     if i < len(input_ids) and i < len(reencoded_ids):
        #         orig_token = tokenizer.decode([input_ids[i]])
        #         renc_token = tokenizer.decode([reencoded_ids[i]]) if i < len(reencoded_ids) else "N/A"
        #         marker = " <-- MISMATCH" if i == mismatch_start else ""
        #         print(f"  {i}: {input_ids[i]}('{orig_token}') vs {reencoded_ids[i] if i < len(reencoded_ids) else 'N/A'}('{renc_token}'){marker}")
    
    # 检查文本级别的差异
    # print(f"原始文本片段: {original_text[max(0, len(original_text)//2-50):len(original_text)//2+50]}")
    # reconstructed_text = tokenizer.decode(reencoded_ids)
    # print(f"重构文本片段: {reconstructed_text[max(0, len(reconstructed_text)//2-50):len(reconstructed_text)//2+50]}")

def ensure_token_consistency(input_ids, tokenizer):
    """
    确保token序列在解码-编码循环后保持一致性
    
    Args:
        input_ids: 原始token序列
        tokenizer: tokenizer实例
        
    Returns:
        一致的token序列
    """
    try:
        # 首先尝试标准解码-编码
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        reencoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
        
        # 检查是否保持一致性
        if len(input_ids) == len(reencoded_ids) and all(a == b for a, b in zip(input_ids, reencoded_ids)):
            # print("标准方法保持token一致性")
            return reencoded_ids
        
        # print(f"警告: 标准方法不一致: {len(input_ids)} -> {len(reencoded_ids)}")
        
        # 分析不匹配原因
        analyze_token_mismatch(input_ids, tokenizer)
        
        # 方法2: 尝试分段处理
        # print("尝试分段处理...")
        chunk_size = 50  # 每次处理50个token
        consistent_chunks = []
        
        for i in range(0, len(input_ids), chunk_size):
            chunk = input_ids[i:i+chunk_size]
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            chunk_reencoded = tokenizer.encode(chunk_text, add_special_tokens=False)
            
            if chunk == chunk_reencoded:
                consistent_chunks.extend(chunk_reencoded)
            else:
                # print(f"分块{i//chunk_size}不一致，使用原始token")
                consistent_chunks.extend(chunk)
        
        if len(consistent_chunks) == len(input_ids) and all(a == b for a, b in zip(input_ids, consistent_chunks)):
            # print("分段方法保持token一致性")
            return consistent_chunks
        
        # 方法3: 逐token处理
        # print("尝试逐token处理...")
        safe_tokens = []
        accumulated_text = ""
        
        for token_id in input_ids:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            accumulated_text += token_text
            safe_tokens.append(token_id)
        
        # 验证重构的文本
        final_reencoded = tokenizer.encode(accumulated_text, add_special_tokens=False)
        
        if len(input_ids) == len(final_reencoded) and all(a == b for a, b in zip(input_ids, final_reencoded)):
            # print("逐token方法保持token一致性")
            return final_reencoded
        else:
            # print("所有方法都无法保持一致性，返回原始token序列")
            # print("这可能是tokenizer的固有特性，建议避免解码-编码循环")
            return input_ids
                
    except Exception as e:
        # print(f"token一致性处理出错: {e}")
        return input_ids

def load_user_data(user_id):
    """
    从reviewer_data_processed_strict加载用户历史和候选项。

    读取单个reviewer文件，将前3个作为历史，其余作为候选项。
    支持"user_XXXX"和"reviewer_XXXX"格式的ID。
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
            f"未找到评论者数据文件。尝试了: {', '.join(candidate_paths)}"
        )

    with open(data_file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    if len(all_data) < 4:
        raise ValueError(
            f"{user_id}的数据不足: 需要至少4个项目（3个历史 + 1个候选），得到了{len(all_data)}个"
        )

    history_data = all_data[:3]
    candidate_data = all_data[3:]

    return history_data, candidate_data

def generate_recommendation_with_cacheblend(user_id):
    """使用缓存混合技术生成推荐"""
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

    # print(f"Loaded candidates count: {len(candidate_data)}")
    # print("Sample candidate:", json.dumps(candidate_data[0], indent=2, ensure_ascii=False)[:200] + "...")

    # 追踪位置
    input_ids, all_chunk_ids = tracker.track_positions(history_prompt, candidate_data, query_prompt)

    # 可视化位置
    tracker.visualize_positions()

    # 验证位置并获取验证结果
    all_matched = tracker.verify_positions()

    # 计算追踪统计信息
    position = []
    for item_info in tracker.get_field_positions():
        for field_name, pos in item_info['fields'].items():
            start = pos['start']
            end = pos['end']
            # print(f"字段 {field_name} 位置: [{start}:{end}]")
            position.extend(list(range(start, end + 1)))

    # 添加查询位置
    query_position = tracker.get_query_position()
    if query_position:
        query_start = query_position['start']
        query_end = query_position['end']
        # print(f"查询位置: [{query_start}:{query_end}]")
        position.extend(list(range(query_start, query_end + 1)))

    # print(f"追踪的token位置: {position}")

    # 获取token IDs - 使用一致的编码参数
    prefix_ids = tokenizer.encode(history_prompt, add_special_tokens=False)
    candidate_ids = []
    for prompt in candidate_prompts:
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        candidate_ids.append(encoded)
    query_ids = tokenizer.encode(query_prompt, add_special_tokens=False)

    # print(f"Candidate count: {len(candidate_ids)}")
    
    # 验证编码一致性
    for i, (original_prompt, encoded_ids) in enumerate(zip(candidate_prompts, candidate_ids)):
        decoded_back = tokenizer.decode(encoded_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        consistency = (original_prompt == decoded_back)
        # print(f"候选项 {i} 编码一致性: {consistency}")
        # if not consistency:
        #     print(f"  原始: {original_prompt[:100]}...")
        #     print(f"  解码: {decoded_back[:100]}...")

    # 初始化缓存收集
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    cache_metadata['collect'] = False
    cache_metadata['check'] = False

    # 组织输入序列
    all_chunk_ids = [prefix_ids] + candidate_ids + [query_ids]

    # 收集KV缓存
    num_layer = 32  # Mistral-7B的层数
    chunk_past_key_values = []

    cache_metadata['collect'] = True



    # def save_token_comparison(ids1, ids2, tokenizer, output_file="token_comparison.txt"):
    #     """保存详细的token比较结果到文件"""
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         f.write("Token Comparison Analysis\n")
    #         f.write("========================\n\n")
            
    #         f.write(f"Original sequence length: {len(ids1)}\n")
    #         f.write(f"Processed sequence length: {len(ids2)}\n\n")
            
    #         f.write("Original prompt decoded:\n")
    #         f.write(f"{tokenizer.decode(ids1)}\n\n")
            
    #         f.write("Processed prompt decoded:\n")
    #         f.write(f"{tokenizer.decode(ids2)}\n\n")
            
    #         f.write("Token-by-token comparison:\n")
    #         f.write("-" * 80 + "\n")
    #         f.write(f"{'Index':<6} {'Original ID':<12} {'Original Text':<25} {'Processed ID':<12} {'Processed Text':<25} {'Match':<6}\n")
    #         f.write("-" * 80 + "\n")
            
    #         max_len = max(len(ids1), len(ids2))
    #         diff_count = 0
            
    #         for idx in range(max_len):
    #             orig_id = ids1[idx] if idx < len(ids1) else "N/A"
    #             proc_id = ids2[idx] if idx < len(ids2) else "N/A"
                
    #             orig_text = tokenizer.decode([orig_id]) if idx < len(ids1) else "N/A"
    #             proc_text = tokenizer.decode([proc_id]) if idx < len(ids2) else "N/A"
                
    #             # 清理文本显示（处理特殊字符）
    #             orig_text_clean = repr(orig_text)[1:-1][:20]
    #             proc_text_clean = repr(proc_text)[1:-1][:20]
                
    #             match = "Y" if orig_id == proc_id else "N"
    #             if orig_id != proc_id:
    #                 diff_count += 1
                
    #             f.write(f"{idx:<6} {str(orig_id):<12} {orig_text_clean:<25} {str(proc_id):<12} {proc_text_clean:<25} {match:<6}\n")
            
    #         f.write("-" * 80 + "\n")
    #         f.write(f"\nSummary:\n")
    #         f.write(f"Total tokens compared: {max_len}\n")
    #         f.write(f"Different tokens: {diff_count}\n")
    #         f.write(f"Match rate: {(max_len - diff_count) / max_len * 100:.2f}%\n")
            
    #         if len(ids1) != len(ids2):
    #             f.write(f"\nLength difference detected:\n")
    #             if len(ids1) > len(ids2):
    #                 f.write(f"Extra tokens in original: {ids1[len(ids2):]}\n")
    #                 f.write(f"Extra tokens decoded: {tokenizer.decode(ids1[len(ids2):])}\n")
    #             else:
    #                 f.write(f"Extra tokens in processed: {ids2[len(ids1):]}\n")
    #                 f.write(f"Extra tokens decoded: {tokenizer.decode(ids2[len(ids1):])}\n")

    # print(f"all_chunk_ids[0]: {all_chunk_ids[0]}")
    
    # 使用一致性保证函数处理token序列
    # ids1 = all_chunk_ids[0]
    # ids2_consistent = ensure_token_consistency(all_chunk_ids[0], tokenizer)
    # ids2_original_method = tokenizer.encode(tokenizer.decode(all_chunk_ids[0]))[1:]
    
    # print(f"Original method processed: {ids2_original_method}")
    # print(f"Consistent method result: {ids2_consistent}")
    
    # 验证一致性方法的效果
    # consistency_check = all(a == b for a, b in zip(ids1, ids2_consistent))
    # print(f"Consistency method works: {consistency_check}")
    
    # 保存详细比较结果到文件（显示原问题）
    # save_token_comparison(ids1, ids2_original_method, tokenizer, "token_comparison_analysis.txt")
    # print(f"Token comparison saved to: token_comparison_analysis.txt")
    
    # print(f"len(ids1): {len(ids1)}, len(ids2_original): {len(ids2_original_method)}")
    
    # 使用真正的token一致性处理方法
    def improved_decode_encode(input_ids, tokenizer):
        """改进的解码-编码方法，真正解决一致性问题"""
        return ensure_token_consistency(input_ids, tokenizer)
    
    # 使用改进方法处理所有chunk
    all_chunk_ids_processed = []
    for i, chunk_ids in enumerate(all_chunk_ids):
        processed_chunk = improved_decode_encode(chunk_ids, tokenizer)
        all_chunk_ids_processed.append(processed_chunk)
        # print(f"Chunk {i} consistency: {len(chunk_ids) == len(processed_chunk) and all(a == b for a, b in zip(chunk_ids, processed_chunk))}")
    
    # 更新all_chunk_ids为处理后的版本
    all_chunk_ids = all_chunk_ids_processed

    # 为每个块收集KV缓存
    gpu_concat_time = 0.0
    for i in range(len(all_chunk_ids)):
        # 直接使用token序列而不是重新解码，避免一致性问题
        if i == 0:
            # 处理前缀 - 使用安全的解码方法
            prompt = tokenizer.decode(all_chunk_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            # 处理候选项和查询 - 使用安全的解码方法  
            prompt = tokenizer.decode(all_chunk_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        # 生成采样参数（只收集KV，不实际生成）
        llm.generate([prompt], SamplingParams(temperature=0.1, max_tokens=1), use_tqdm=False)

        # 从模型获取每层的KV缓存
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers

        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv

            if i == 0:
                # 处理前缀
                temp_k = past_key_values[0].clone()
                temp_v = past_key_values[1].clone()
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                # 处理后续内容，跳过BOS token
                temp_k = past_key_values[0][1:].clone()
                temp_v = past_key_values[1][1:].clone()

                # 连接KV缓存
                concat_start = time.time()
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_v), dim=0)
                gpu_concat_time += time.time() - concat_start

        # print(f"处理块 {i}，形状: {temp_k.shape[0]}")

    # 设置合并的KV缓存
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
    # print(f"最终KV缓存形状: {chunk_past_key_values[0][0].shape[0]}")

    # 构建完整输入 - 直接连接token序列，保持一致性
    input_ids = []
    for i in range(len(all_chunk_ids)):
        input_ids.extend(all_chunk_ids[i])

    # print(f"完整input_ids长度: {len(input_ids)}")

    # 边界检查：确保position中的索引不会溢出input_ids的长度
    input_ids_length = len(input_ids)
    position = [idx for idx in position if 0 <= idx < input_ids_length]
    # print(f"边界检查后的position长度: {len(position)}")

    # 使用安全的解码方法
    input_prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # 使用缓存生成
    cache_metadata["check"] = True
    cache_metadata['collect'] = False
    # cache_metadata['suffix_len'] = len(query_ids)
    cache_metadata['imp_indices'] = position
    # cache_metadata['imp_indices'] = list(range(len(input_ids)))

    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    output = llm.generate([input_prompt], sampling_params, use_tqdm=False)

    # print(f"Generated with cache: {output[0].outputs[0].text}")
    gpu_cache_ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"GPU Cache TTFT: {gpu_cache_ttft}")
    print(f"GPU Concat Time: {gpu_concat_time:.6f}s")
    print("------------")

    cache_metadata["check"] = False
    cache_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params, use_tqdm=False)
    # print(f"Normal generation: {output[0].outputs[0].text}")
    full_prefill_ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"Full prefill TTFT: {full_prefill_ttft}")
    print("------------")
    
    return {
        'gpu_cache_ttft': gpu_cache_ttft,
        'full_prefill_ttft': full_prefill_ttft,
        'gpu_concat_time': gpu_concat_time
    }

def generate_recommendation_with_cpu_cache(user_id):
    """使用CPU存储的KV缓存技术生成推荐"""
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

    # print(f"Loaded candidates count: {len(candidate_data)}")
    # print("Sample candidate:", json.dumps(candidate_data[0], indent=2, ensure_ascii=False)[:200] + "...")

    # 追踪位置
    input_ids, all_chunk_ids = tracker.track_positions(history_prompt, candidate_data, query_prompt)

    # 可视化位置
    tracker.visualize_positions()

    # 验证位置并获取验证结果
    all_matched = tracker.verify_positions()

    # 计算追踪统计信息
    position = []
    for item_info in tracker.get_field_positions():
        for field_name, pos in item_info['fields'].items():
            start = pos['start']
            end = pos['end']
            position.extend(list(range(start, end + 1)))

    # 添加查询位置
    query_position = tracker.get_query_position()
    if query_position:
        query_start = query_position['start']
        query_end = query_position['end']
        position.extend(list(range(query_start, query_end + 1)))

    # 获取token IDs - 使用一致的编码参数
    prefix_ids = tokenizer.encode(history_prompt, add_special_tokens=False)
    candidate_ids = []
    for prompt in candidate_prompts:
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        candidate_ids.append(encoded)
    query_ids = tokenizer.encode(query_prompt, add_special_tokens=False)

    # print(f"Candidate count: {len(candidate_ids)}")
    
    # 验证编码一致性
    for i, (original_prompt, encoded_ids) in enumerate(zip(candidate_prompts, candidate_ids)):
        decoded_back = tokenizer.decode(encoded_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        consistency = (original_prompt == decoded_back)

    # 初始化缓存收集
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    cache_metadata['collect'] = False
    cache_metadata['check'] = False

    # 组织输入序列
    all_chunk_ids = [prefix_ids] + candidate_ids + [query_ids]

    # 收集KV缓存 - CPU版本
    num_layer = 32  # Mistral-7B的层数
    chunk_past_key_values_cpu = []

    cache_metadata['collect'] = True

    # 使用改进方法处理所有chunk
    all_chunk_ids_processed = []
    for i, chunk_ids in enumerate(all_chunk_ids):
        processed_chunk = ensure_token_consistency(chunk_ids, tokenizer)
        all_chunk_ids_processed.append(processed_chunk)
    
    # 更新all_chunk_ids为处理后的版本
    all_chunk_ids = all_chunk_ids_processed

    # print("=== Starting CPU KV Cache Collection ===")
    
    # Collect KV cache for each chunk and store in CPU
    cpu_concat_time = 0.0
    gpu_to_cpu_time = 0.0
    cpu_to_gpu_time = 0.0
    for i in range(len(all_chunk_ids)):
        # 直接使用token序列而不是重新解码，避免一致性问题
        if i == 0:
            # 处理前缀 - 使用安全的解码方法
            prompt = tokenizer.decode(all_chunk_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            # 处理候选项和查询 - 使用安全的解码方法  
            prompt = tokenizer.decode(all_chunk_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        # 生成采样参数（只收集KV，不实际生成）
        llm.generate([prompt], SamplingParams(temperature=0.1, max_tokens=1), use_tqdm=False)

        # 从模型获取每层的KV缓存
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers

        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv

            if i == 0:
                # 处理前缀 - 存储到CPU
                transfer_start = time.time()
                temp_k = past_key_values[0].clone().cpu()  # 移动到CPU
                temp_v = past_key_values[1].clone().cpu()  # 移动到CPU
                gpu_to_cpu_time += time.time() - transfer_start
                chunk_past_key_values_cpu.append([temp_k, temp_v])
                # print(f"Prefix chunk GPU->CPU: K shape {past_key_values[0].shape}, V shape {past_key_values[1].shape}")
            else:
                # 处理后续内容，跳过BOS token - 存储到CPU
                transfer_start = time.time()
                temp_k = past_key_values[0][1:].clone().cpu()  # 移动到CPU
                temp_v = past_key_values[1][1:].clone().cpu()  # 移动到CPU
                gpu_to_cpu_time += time.time() - transfer_start

                # 在CPU上连接KV缓存
                concat_start = time.time()
                chunk_past_key_values_cpu[j][0] = torch.cat((chunk_past_key_values_cpu[j][0], temp_k), dim=0)
                chunk_past_key_values_cpu[j][1] = torch.cat((chunk_past_key_values_cpu[j][1], temp_v), dim=0)
                cpu_concat_time += time.time() - concat_start
                # print(f"Chunk {i} GPU->CPU: Original shape {past_key_values[0].shape}, After slicing {temp_k.shape}")

        # print(f"Processing chunk {i}, CPU cache shape: K{chunk_past_key_values_cpu[0][0].shape}, V{chunk_past_key_values_cpu[0][1].shape}")

    # print("=== CPU KV Cache Collection Complete ===")

    # Move CPU cache back to GPU for inference
    # print("=== Moving CPU Cache Back to GPU ===")
    chunk_past_key_values_gpu = []
    transfer_start = time.time()
    for j in range(num_layer):
        gpu_k = chunk_past_key_values_cpu[j][0].cuda()  # CPU -> GPU
        gpu_v = chunk_past_key_values_cpu[j][1].cuda()  # CPU -> GPU
        chunk_past_key_values_gpu.append([gpu_k, gpu_v])
        # print(f"Layer {j}: CPU cache moved back to GPU, K shape {gpu_k.shape}, V shape {gpu_v.shape}")
    cpu_to_gpu_time = time.time() - transfer_start

    # Set the merged KV cache
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values_gpu
    # print(f"Final GPU KV cache shape: {chunk_past_key_values_gpu[0][0].shape[0]}")

    # 构建完整输入 - 直接连接token序列，保持一致性
    input_ids = []
    for i in range(len(all_chunk_ids)):
        input_ids.extend(all_chunk_ids[i])

    # 边界检查：确保position中的索引不会溢出input_ids的长度
    input_ids_length = len(input_ids)
    position = [idx for idx in position if 0 <= idx < input_ids_length]

    # 使用安全的解码方法
    input_prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # 使用缓存生成
    cache_metadata["check"] = True
    cache_metadata['collect'] = False
    cache_metadata['imp_indices'] = position

    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    output = llm.generate([input_prompt], sampling_params, use_tqdm=False)

    cpu_cache_ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"CPU Cache TTFT: {cpu_cache_ttft}")
    print(f"GPU->CPU Transfer Time: {gpu_to_cpu_time:.6f}s")
    print(f"CPU Concat Time: {cpu_concat_time:.6f}s")
    print(f"CPU->GPU Transfer Time: {cpu_to_gpu_time:.6f}s")
    total_cpu_overhead = cpu_to_gpu_time + cpu_cache_ttft
    print(f"Total CPU Overhead: {total_cpu_overhead:.6f}s")
    print("------------")

    cache_metadata["check"] = False
    cache_metadata['collect'] = False
    
    return {
        'cpu_cache_ttft': cpu_cache_ttft,
        'gpu_to_cpu_time': gpu_to_cpu_time,
        'cpu_concat_time': cpu_concat_time,
        'cpu_to_gpu_time': cpu_to_gpu_time,
        'total_cpu_overhead': total_cpu_overhead
    }

# def test_token_consistency():
#     """测试token一致性修复效果"""
#     # print("\n=====测试Token一致性修复效果=====")
    
#     # 使用从分析文件中发现有问题的文本进行测试
#     problematic_text = "You are an intelligent assistant that can rank items based on the user's preference.\nUser A10FW892S59ABJ has purchased the following items in this order:\n- {'itemID': 'B002NGMEQC', 'title': 'BaByliss Pro BABCHV21 Ceramic Instant Heat 20-Roller Set, Varied','salesRank': {'Beauty': 95234}, 'categories': [['Beauty', 'Hair Care', 'Styling Tools', 'Hair Rollers']], 'price': nan, 'brand': nan}\n- {'itemID': 'B004VMGTY4', 'title': \"Burts Bees Sensitive Daily Moisturizing Cream, 1.8 Ounces\",'salesRank': {'Beauty': 626}, 'categories': [['Beauty', 'Skin Care', 'Face', 'Creams & Moisturizers']], 'price': 8.35, 'brand': 'Burts Bees'}"
    
#     # print("使用实际问题文本进行测试...")
#     # print(f"测试文本长度: {len(problematic_text)}")
    
#     # 编码原始文本
#     original_ids = tokenizer.encode(problematic_text, add_special_tokens=False)
#     # print(f"原始token数量: {len(original_ids)}")
    
#     # 原始的有问题方法
#     original_decoded = tokenizer.decode(original_ids)
#     original_reencoded = tokenizer.encode(original_decoded, add_special_tokens=False)
    
#     # print(f"原始方法重编码数量: {len(original_reencoded)}")
#     # print(f"原始方法一致性: {original_ids == original_reencoded}")
    
#     # 使用改进的一致性保证方法
#     # print("\n--- 使用改进的ensure_token_consistency方法 ---")
#     consistent_ids = ensure_token_consistency(original_ids, tokenizer)
    
#     # print(f"改进方法结果数量: {len(consistent_ids)}")
#     final_consistency = (original_ids == consistent_ids)
#     # print(f"改进方法一致性: {final_consistency}")
    
#     if not final_consistency:
#         # print("分析差异...")
#         # diff_count = 0
#         # for i, (orig, new) in enumerate(zip(original_ids, consistent_ids)):
#         #     if orig != new:
#         #         diff_count += 1
#         #         if diff_count <= 5:  # 只显示前5个差异
#         #             print(f"  位置{i}: {orig} -> {new}")
#         # print(f"总共{diff_count}个差异")
#         pass
    
#     # 测试其他类型的文本
#     test_cases = [
#         "简单英文测试",
#         "包含HTML实体测试",
#         "Burts Bees",
#         json.dumps({"brand": "Burts Bees"}, ensure_ascii=False)
#     ]
    
#     # print("\n--- 测试其他文本类型 ---")
#     success_count = 0
    
#     for i, text in enumerate(test_cases):
#         # print(f"\n测试{i+1}: {text}")
        
#         ids = tokenizer.encode(text, add_special_tokens=False)
#         consistent_result = ensure_token_consistency(ids, tokenizer)
        
#         is_consistent = (ids == consistent_result)
#         # print(f"  结果: {'一致' if is_consistent else '不一致'}")
        
#         if is_consistent:
#             success_count += 1
    
#     print(f"\n=== Test Summary ===")
#     print(f"Success Rate: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    user_id = "user_A10FW892S59ABJ"
    
    print("\n===== KV Cache Performance Data Collection =====")
    
    # Test GPU Cache version
    print("\n=== GPU Cache Version ===")
    gpu_results = generate_recommendation_with_cacheblend(user_id)
    
    # Test CPU Cache version
    print("\n=== CPU Cache Version ===")
    cpu_results = generate_recommendation_with_cpu_cache(user_id)
    
    print("\n===== Collected Performance Data =====")
    print(f"1. Full prefill TTFT: {gpu_results['full_prefill_ttft']:.6f}s")
    print(f"2. GPU Cache TTFT: {gpu_results['gpu_cache_ttft']:.6f}s")
    print(f"3. CPU Cache TTFT: {cpu_results['cpu_cache_ttft']:.6f}s")
    print(f"4. Total CPU Overhead: {cpu_results['total_cpu_overhead']:.6f}s")
    
    print("\n=== Performance Analysis ===")
    gpu_speedup = (gpu_results['full_prefill_ttft'] - gpu_results['gpu_cache_ttft']) / gpu_results['full_prefill_ttft'] * 100
    cpu_speedup = (gpu_results['full_prefill_ttft'] - cpu_results['cpu_cache_ttft']) / gpu_results['full_prefill_ttft'] * 100
    cpu_vs_gpu = (gpu_results['gpu_cache_ttft'] - cpu_results['cpu_cache_ttft']) / gpu_results['gpu_cache_ttft'] * 100
    
    print(f"GPU Cache vs Full prefill: {gpu_speedup:.2f}% improvement")
    print(f"CPU Cache vs Full prefill: {cpu_speedup:.2f}% improvement")
    print(f"CPU Cache vs GPU Cache: {cpu_vs_gpu:.2f}% {'improvement' if cpu_vs_gpu > 0 else 'degradation'}")
    
    print("\n=== Data Summary (CSV Format) ===")
    print("Metric,Time(s)")
    print(f"Full_prefill_TTFT,{gpu_results['full_prefill_ttft']:.6f}")
    print(f"GPU_Cache_TTFT,{gpu_results['gpu_cache_ttft']:.6f}")
    print(f"CPU_Cache_TTFT,{cpu_results['cpu_cache_ttft']:.6f}")
    print(f"Total_CPU_Overhead,{cpu_results['total_cpu_overhead']:.6f}")
