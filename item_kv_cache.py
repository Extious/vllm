import os
import json
import torch
import time
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 设置环境变量
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

class ItemKVCache:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name = model_name
        print(f"正在加载模型 {model_name}...")
        self.llm = LLM(model=model_name, gpu_memory_utilization=0.95)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            config = self.llm.llm_engine.model_config.hf_config
            self.num_layers = config.num_hidden_layers
            print(f"检测到层数: {self.num_layers}")
        except Exception:
            self.num_layers = 32
            print(f"无法自动检测层数，使用默认值: {self.num_layers}")
        self.kv_cache = {}  # CPU内存中的KV缓存

    def collect_item_kv(self, item):
        item_id = item.get("itemID")
        if not item_id:
            return
        prompt = str(item)
        prompt_token_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)
        try:
            cache_metadata = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata
            cache_metadata['collect'] = True
            cache_metadata['check'] = False
            llm_layers = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
            for layer in llm_layers:
                layer.self_attn.hack_kv = None
            self.llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1))
            item_kv_cache_gpu = [None] * self.num_layers
            for layer_idx in range(self.num_layers):
                full_past_key_values = llm_layers[layer_idx].self_attn.hack_kv
                if full_past_key_values is None or len(full_past_key_values) != 2:
                    return
                current_k = full_past_key_values[0][:prompt_len].clone()
                current_v = full_past_key_values[1][:prompt_len].clone()
                item_kv_cache_gpu[layer_idx] = [current_k, current_v]
            item_kv_cache_cpu = [
                [k.cpu().to(torch.bfloat16), v.cpu().to(torch.bfloat16)]
                for k, v in item_kv_cache_gpu
            ]
            self.kv_cache[item_id] = item_kv_cache_cpu
            # 存储到硬盘
            save_path = os.path.join(self.output_dir, f"kv_{item_id}.pt")
            torch.save(item_kv_cache_cpu, save_path)
            for layer in llm_layers:
                layer.self_attn.hack_kv = None
            cache_metadata['collect'] = False
        except Exception as e:
            print(f"收集item {item_id}的KV缓存失败: {e}")

    def get_item_kv(self, item_id):
        return self.kv_cache.get(item_id, None)

if __name__ == "__main__":
    items_file_path = "../all_items.json"
    output_dir = "../kv_cache_cpu"  # 存储KV缓存的目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"从 {items_file_path} 加载items...")
    with open(items_file_path, 'r') as f:
        all_items = json.load(f)
    all_items = [item for item in all_items if isinstance(item, dict) and "itemID" in item]
    print(f"共加载 {len(all_items)} 个item。")

    cache_builder = ItemKVCache()
    cache_builder.output_dir = output_dir

    # 计算并存储所有item的KV缓存
    build_start = time.time()
    for i, item in enumerate(all_items):
        cache_builder.collect_item_kv(item)
        if (i+1) % 10 == 0 or i == len(all_items)-1:
            print(f"已处理 {i+1}/{len(all_items)} 个item...")
    build_end = time.time()
    print(f"所有item的KV缓存已收集并存储到CPU和硬盘，用时: {build_end-build_start:.2f}秒")

    # 测试itemID查询时间
    test_ids = [item["itemID"] for item in all_items[:min(100, len(all_items))]]  # 取前100个itemID测试
    retrieval_times = []
    for item_id in test_ids:
        t0 = time.time()
        kv = cache_builder.get_item_kv(item_id)
        t1 = time.time()
        retrieval_times.append(t1-t0)
    print("\nitemID查询KV缓存耗时统计（单位: 秒）：")
    print("itemID      | 查询耗时")
    print("------------|----------")
    for i, item_id in enumerate(test_ids):
        print(f"{item_id:<12}| {retrieval_times[i]:.6f}")
    print(f"\n平均查询耗时: {np.mean(retrieval_times):.6f} 秒")
    print("脚本执行完毕。")