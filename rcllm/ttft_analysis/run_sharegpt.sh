#!/bin/bash

# 确保本地连接不走代理
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,localhost"

# 设置 HuggingFace 离线模式，避免网络访问
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 检查模型缓存目录是否存在
MODEL_CACHE_DIR="$HOME/.cache/huggingface/hub"
if [ ! -d "$MODEL_CACHE_DIR" ]; then
    echo "Warning: HuggingFace cache directory not found at $MODEL_CACHE_DIR"
    echo "You may need to download the model first or disable offline mode"
fi

python3 benchmarks/benchmark_serving.py \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --tokenizer-mode mistral \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /home/comp/24481750/vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 \
  --max-concurrency 10 \
  --request-rate 2
