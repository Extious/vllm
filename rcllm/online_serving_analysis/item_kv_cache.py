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
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        print(f"Loading model {model_name}...")
        self.llm = LLM(model=model_name, gpu_memory_utilization=0.95)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            config = self.llm.llm_engine.model_config.hf_config
            self.num_layers = config.num_hidden_layers
            print(f"Detected layers: {self.num_layers}")
        except Exception:
            self.num_layers = 32
            print(f"Unable to auto-detect layers, using default: {self.num_layers}")
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
            print(f"Failed to collect KV cache for item {item_id}: {e}")

    def get_item_kv(self, item_id):
        return self.kv_cache.get(item_id, None)

if __name__ == "__main__":
    items_file_path = "../all_items.json"
    output_dir = "../kv_cache_cpu"  # 存储KV缓存的目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading items from {items_file_path}...")
    with open(items_file_path, 'r') as f:
        all_items = json.load(f)
    all_items = [item for item in all_items if isinstance(item, dict) and "itemID" in item]
    print(f"Loaded {len(all_items)} items in total.")

    cache_builder = ItemKVCache()
    cache_builder.output_dir = output_dir

    # 计算并存储所有item的KV缓存
    build_start = time.time()
    for i, item in enumerate(all_items):
        cache_builder.collect_item_kv(item)
        if (i+1) % 10 == 0 or i == len(all_items)-1:
            print(f"Processed {i+1}/{len(all_items)} items...")
    build_end = time.time()
    print(f"All item KV caches collected and stored to CPU and disk, time taken: {build_end-build_start:.2f} seconds")

    # 测试itemID查询时间
    test_ids = [item["itemID"] for item in all_items[:min(100, len(all_items))]]  # 取前100个itemID测试
    retrieval_times = []
    for item_id in test_ids:
        t0 = time.time()
        kv = cache_builder.get_item_kv(item_id)
        t1 = time.time()
        retrieval_times.append(t1-t0)
    print("\nItemID KV cache query time statistics (unit: seconds):")
    print("itemID      | Query Time")
    print("------------|----------")
    for i, item_id in enumerate(test_ids):
        print(f"{item_id:<12}| {retrieval_times[i]:.6f}")
    print(f"\nAverage query time: {np.mean(retrieval_times):.6f} seconds")
    print("Script execution completed.")