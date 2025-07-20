#!/bin/bash

# 简单解决方案：使用本地已缓存的模型
# 确保本地连接不走代理
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,localhost"

# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Running ShareGPT benchmark with offline mode..."
echo "Make sure vLLM server is running at http://127.0.0.1:8000"

# 检查服务器状态
if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server is not running!"
    echo "Please start it with: CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Mistral-7B-Instruct-v0.3 --gpu-memory-utilization 0.8"
    exit 1
fi

echo "Server is running, starting benchmark..."

# 使用本地模型路径（如果存在）或者依赖服务器端的tokenizer
python3 benchmarks/benchmark_serving.py \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --tokenizer-mode mistral \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /home/comp/24481750/vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 \
  --max-concurrency 10 \
  --request-rate 2 \
  --trust-remote-code
