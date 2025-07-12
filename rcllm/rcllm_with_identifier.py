import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
import logging
from prompt_tracker_with_identifier import PromptFieldTracker

# --- 日志记录配置 ---
DEBUG_MODE = True  # True 表示调试模式，False 表示生产模式
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

# 初始化大模型
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95,enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# llm.set_tokenizer(tokenizer) # set_tokenizer is deprecated

def load_user_data(user_id):
    """加载用户历史购买数据和候选商品"""
    user_dir = os.path.join(os.path.dirname(__file__),"../../dataset", user_id)
    
    # 读取历史购买记录
    history_path = os.path.join(user_dir, "history.json")
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    # 读取候选商品
    candidate_path = os.path.join(user_dir, "candidate.json")
    with open(candidate_path, 'r') as f:
        candidate_data = json.load(f)
        
    return history_data, candidate_data

def generate_recommendation_with_cache(user_id):
    # 加载用户数据
    history_data, candidate_data = load_user_data(user_id)
    
    # 提取用户名（从用户ID）
    username = user_id.split('_')[-1] # username not used in prefix
    
    # 构建基本提示前缀
    prefix_prompt = f"""You are an intelligent assistant that can rank items based on the user's preference. The history items and candidate items are listed below. The prefix of history items should be [history] and the prefix of candidate items should be [i]. i is the identifier of the candidate item. Please rank the candidate items based on the user's history. You should strictly obey the following rules: 
    1. All the candidate items should be included and listed using identifiers, in descending order of the user's preference. The most preferred recommendation item should be listed first.
    2. The results format should be [] > [], where each [] is an identifier, e.g., [2] > [1] > [0].
    3. Only respond with the ranking results, do not say any word or explain.
    4. Output in the following JSON format: \n{{\"rank\": \"[] > [] .. > []\"}}."""

    # 将历史数据格式化为提示的一部分 - 这部分逻辑会移到 track_positions 或作为其输入
    # purchase_history = "\n".join([f"- {item}" for item in history_data])
    # history_prompt_for_tracker = f"{prefix}\n{purchase_history}\n\n" # This was the old history_prompt input
    
    # 为每个候选商品创建单独的提示 - 这部分逻辑会移到 track_positions 或作为其输入
    # candidate_prompts_for_tracker = []
    # for i, item in enumerate(candidate_data):
    #     candidate_prompt = f"[{i}]: {item}" # item here is directly the candidate, not json.dumps(item)
    #     candidate_prompts_for_tracker.append(candidate_prompt) # This seems to be based on old structure.
                                                          # New track_positions takes raw candidate_data (list of dicts)
    
    # 创建查询提示
    query_prompt = f"""\n{len(history_data)} history items and {len(candidate_data)} candidate items are listed above. Be careful that the number of the ranking items is {len(candidate_data)}. Please give me the JSON format data of the ranking results, do not output anything other than the JSON format data."""
    
    # 创建PromptFieldTracker的实例
    tracker = PromptFieldTracker(tokenizer)
    
    logger.info(f"Number of loaded candidates: {len(candidate_data)}")
    logger.info(f"Example candidate: {json.dumps(candidate_data[0], indent=2)[:200]}...")
    
    # 跟踪位置
    # input_ids, all_chunk_ids = tracker.track_positions(history_prompt_for_tracker, candidate_data, query_prompt)
    all_chunk_ids, input_ids, value_positions, query_position = tracker.track_positions(
        prefix_prompt, 
        history_data, 
        candidate_data, 
        query_prompt
    )

    # 验证query_position的最后一个整数是否在input_ids范围内
    if query_position:
        assert query_position[-1] < len(input_ids), \
            f"Query position end {query_position[-1]} out of bounds (input_ids length: {len(input_ids)})"
    
    
    # 计算跟踪统计信息
    imp_indices = []
    imp_indices.extend(value_positions)
    imp_indices.extend(query_position)

    # imp_indices = list(range(len(input_ids)+1))

    logger.debug(f"Tracked token positions from candidate values: {len(imp_indices)} tokens, {imp_indices}")
    logger.info(f"Number of chunks in all_chunk_ids: {len(all_chunk_ids)}")


    
    # 初始化缓存收集
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
    cache_metadata['collect'] = False
    cache_metadata['check'] = False
    
    # 收集KV缓存
    num_layer = 32  # Mistral-7B中的层数
    chunk_past_key_values = []
    
    cache_metadata['collect'] = True
    
    # 为每个块收集KV缓存
    for i in range(len(all_chunk_ids)):
        if i == 0:
            # 处理前缀 + 历史
            prompt_text_for_kv_collection = tokenizer.decode(all_chunk_ids[i])
        else:
            # 处理候选商品和查询
            prompt_text_for_kv_collection = tokenizer.decode(all_chunk_ids[i]) # BOS is already removed from chunks
        
        # 生成采样参数（仅收集KV，不实际生成）
        # We need to pass the text that corresponds to the token IDs in all_chunk_ids[i]
        # Since BOS was removed, we should decode and pass that.
        # However, llm.generate expects a prompt that would be tokenized with a BOS.
        # The original code did `llm.generate([prompt], ...)` where `prompt` was a string.
        # If `all_chunk_ids[i]` are tokens *without* BOS, decoding them and then re-tokenizing in `generate`
        # might lead to slight differences if not handled carefully.
        # The original code's loop for KV collection:
        # `prompt = tokenizer.decode(all_chunk_ids[i])` - this `all_chunk_ids[i]` was BOS-less.
        # `llm.generate([prompt], ...)` - this `prompt` string would be tokenized by vLLM, likely adding BOS.
        # This seems to have been the working model.
        
        llm.generate([prompt_text_for_kv_collection], SamplingParams(temperature=0.1, max_tokens=1))
        
        # 从模型的每一层获取KV缓存
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            
            if i == 0:
                # 处理前缀
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
                # 处理后续内容，跳过BOS令牌
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
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_k), dim=0)
        
        logger.debug(f"Processed KV for chunk {i}, num_tokens in chunk (BOS-less): {len(all_chunk_ids[i])}, KV cache collected for {past_key_values[0].shape[0]} tokens (with BOS)")
    
    # 设置合并的KV缓存
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
    logger.info(f"Final KV cache shape: {chunk_past_key_values[0][0].shape[0]}")
    
    
    # input_prompt should be decodable from input_ids_concat, which is same as self.input_ids from tracker
    input_prompt = tokenizer.decode(input_ids) # Use input_ids from tracker directly
    
    # 使用缓存生成
    cache_metadata["check"] = True
    cache_metadata['collect'] = False
    # cache_metadata['suffix_len'] = len(query_ids) # 后缀长度
    cache_metadata['imp_indices'] = imp_indices

    print(f"input_ids_len: {len(input_ids)}")
    print(f"imp_indices_len: {len(cache_metadata['imp_indices'])}")
    print(f"First 10 imp_indices: {cache_metadata['imp_indices'][:10] if cache_metadata['imp_indices'] else 'None'}")
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    output = llm.generate([input_prompt], sampling_params)
    
    print(f"Generation with cache: {output[0].outputs[0].text}")
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")
    # return output[0].outputs[0].text # 返回输出文本

    cache_metadata["check"] = False
    cache_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")

if __name__ == "__main__":
    user_id = "user_A10FW892S59ABJ"
    logger.info("\n=====Generating Recommendations=====")
    # recommendation = generate_recommendation_with_cache(user_id) # Function doesn't return anymore
    generate_recommendation_with_cache(user_id)
    logger.info("===== End =====")