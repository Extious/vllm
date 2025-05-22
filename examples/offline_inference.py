import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTHONIOENCODING"] = "utf-8"
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct",gpu_memory_utilization=0.95,pipeline_parallel_size=2)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
