import os

os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["VLLM_USE_V1"] = "0"
os.environ["PYTHONIOENCODING"] = "UTF-8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen3-32B", gpu_memory_utilization=0.8, tensor_parallel_size=4, enable_chunked_prefill=False)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


import contextlib
import torch
import time

# 添加同步等待，确保所有进程完成
if torch.distributed.is_initialized():
    torch.distributed.barrier()
    time.sleep(1)  # 给进程间通信一些缓冲时间

# 按正确顺序清理分布式环境
try:
    destroy_model_parallel()
    destroy_distributed_environment()
except Exception as e:
    print(f"Warning during cleanup: {e}")

# 最后清理torch分布式进程组
with contextlib.suppress(Exception):
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()