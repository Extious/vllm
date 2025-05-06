import os
import torch
import time
import numpy as np

if __name__ == "__main__":
    kv_dir = "../kv_cache_cpu"  # KV缓存存储目录
    print(f"从 {kv_dir} 加载所有item的KV...")
    kv_files = [f for f in os.listdir(kv_dir) if f.startswith("kv_") and f.endswith(".pt")]
    print(f"共检测到 {len(kv_files)} 个KV文件。")

    # 1. 加载所有KV到CPU内存，并建立itemID索引
    kv_cache = {}
    for i, fname in enumerate(kv_files):
        item_id = fname[len("kv_"):-len(".pt")]
        kv_path = os.path.join(kv_dir, fname)
        kv = torch.load(kv_path, map_location="cpu")
        kv_cache[item_id] = kv
        if (i+1) % 10 == 0 or i == len(kv_files)-1:
            print(f"已加载 {i+1}/{len(kv_files)} 个KV...")
    print(f"所有KV已加载到CPU内存。")

    # 2. 测试itemID查询并转到GPU的耗时
    test_ids = list(kv_cache.keys())[:min(100, len(kv_cache))]  # 取前100个itemID测试
    retrieval_times = []
    for item_id in test_ids:
        t0 = time.time()
        kv_cpu = kv_cache[item_id]
        # 转到GPU
        kv_gpu = [[k.cuda().to(torch.bfloat16), v.cuda().to(torch.bfloat16)] for k, v in kv_cpu]
        print(kv_gpu[0][0].shape)
        t1 = time.time()
        retrieval_times.append(t1-t0)
        # 释放显存
        del kv_gpu
        torch.cuda.empty_cache()
    print("\nitemID查询+转GPU耗时统计（单位: 秒）：")
    print("itemID      | 查询+转GPU耗时")
    print("------------|----------------")
    for i, item_id in enumerate(test_ids):
        print(f"{item_id:<12}| {retrieval_times[i]:.6f}")
    print(f"\n平均查询+转GPU耗时: {np.mean(retrieval_times):.6f} 秒")
    print("脚本执行完毕。") 