# SPDX-License-Identifier: Apache-2.0
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse
import os
import sys
import time

import requests
from openai import OpenAI

# Set no_proxy to bypass proxy for localhost connections
os.environ['no_proxy'] = '127.0.0.1,localhost'
os.environ['NO_PROXY'] = '127.0.0.1,localhost'  # Some libraries check uppercase version

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
    {"role": "user", "content": "Where was it played?"},
]


def test_server_connection(base_url: str, timeout: int = 5) -> bool:
    """Test if the vLLM server is running and accessible."""
    try:
        # Try to connect to the health endpoint
        health_url = base_url.replace("/v1", "/health")
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass

    try:
        # Fallback: try to connect to the models endpoint
        models_url = f"{base_url}/models"
        response = requests.get(models_url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum number of retries (default: 3)"
    )
    return parser.parse_args()


def main(args):
    print(f"Connecting to vLLM server at: {openai_api_base}")
    print(f"Timeout: {args.timeout} seconds")
    print(f"Max retries: {args.max_retries}")
    print("-" * 50)

    # 首先测试服务器连接
    print("Testing server connection...")
    if not test_server_connection(openai_api_base, timeout=5):
        print(f"ERROR: Cannot connect to vLLM server at {openai_api_base}")
        print("Please ensure the server is running with:")
        print("  vllm serve <model_name> --port 8000")
        print("Or check if the server is running on a different port.")
        sys.exit(1)

    print("Server connection successful!")

    # 创建带超时设置的OpenAI客户端
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=args.timeout,
    )

    # 获取可用模型，支持重试逻辑
    for attempt in range(args.max_retries):
        try:
            print(f"Fetching available models... (attempt {attempt + 1}/{args.max_retries})")
            models = client.models.list()
            if not models.data:
                print("ERROR: No models available on the server")
                sys.exit(1)

            model = models.data[0].id
            print(f"Using model: {model}")
            break

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == args.max_retries - 1:
                print("Failed to fetch models after all retries")
                sys.exit(1)
            print(f"Retrying in 2 seconds...")
            time.sleep(2)

    # 聊天完成API调用，包含错误处理
    try:
        print("Sending chat completion request...")
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            stream=args.stream,
        )

        print("-" * 50)
        print("Chat completion results:")
        if args.stream:
            try:
                for chunk in chat_completion:
                    print(chunk)
            except Exception as e:
                print(f"ERROR: Error during streaming: {e}")
                sys.exit(1)
        else:
            print(chat_completion)
        print("-" * 50)
        print("Chat completion successful!")

    except Exception as e:
        print(f"ERROR: Error during chat completion: {e}")
        print("This might be due to:")
        print("  - Model not loaded properly")
        print("  - Server overloaded")
        print("  - Network timeout")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
