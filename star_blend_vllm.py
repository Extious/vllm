import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.5,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm.set_tokenizer(tokenizer)

#TODO (Jiayi): fix last len


f = open(f"star_input/prompt_v1.json")
ex = json.load(f)
chunk_num = ex['chunk_num']
doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
q_prompt = ex['query']
doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
q_ids = tokenizer.encode(q_prompt)[1:]

chunk_lengths = [len(chunk) for chunk in doc_chunk_ids]
avg_chunk_size = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0

print(f"doc_chunk_ids: {len(doc_chunk_ids)}")
print(f"chunk_lengths: {chunk_lengths}")


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1)

prefix = ex['prefix']
prefix_ids = tokenizer.encode(prefix)[1:]
prefix_len = len(prefix_ids) + 1


doc_chunk_ids = [chunk_ids for chunk_ids in doc_chunk_ids]
doc_chunk_ids = [prefix_ids] + doc_chunk_ids
doc_chunk_ids = doc_chunk_ids + [q_ids]

    
        
input_ids = []

for i in range(len(doc_chunk_ids)):
    if i == 0:
        temp_ids = doc_chunk_ids[i]
    else:
        temp_ids = doc_chunk_ids[i][:]
    input_ids += temp_ids
        
input_prompt = tokenizer.decode(input_ids)

    
sampling_params = SamplingParams(temperature=0, max_tokens=100)

output = llm.generate([input_prompt], sampling_params)
print(f"Normal generation: {output[0].outputs[0].text}")
print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
print("------------")