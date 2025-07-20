#!/bin/bash
ulimit -n 4096
export no_proxy="127.0.0.1,localhost"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 benchmarks/benchmark_serving.py \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --endpoint /v1/completions \
  --dataset-name random \
  --dataset-path synthetic \
  --save-result \
  --save-detailed \
  --result-filename ttft_wide_dist.json \
  --num-prompts 1000 \
  --max-concurrency 20 \
  --request-rate 20 \
  --random-input-len 16000 \
  --random-output-len 128 \
  --random-range-ratio 0.8
