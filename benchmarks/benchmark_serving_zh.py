# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.
# 基准测试在线服务吞吐量。

On the server side, run one of the following commands:
# 在服务器端，运行以下命令之一：
    vLLM OpenAI API server
    # vLLM OpenAI API 服务器
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

On the client side, run:
# 在客户端，运行：
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
                                         # 默认情况下 <request_rate> 是 inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000
                                    # 默认情况下 <num_prompts> 是 1000

    when using tgi backend, add
    # 当使用 tgi 后端时，在上述命令末尾添加
        --endpoint /generate_stream
    to the end of the command above.
"""

import argparse
import asyncio
import gc
import json
import os
import random
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_dataset import (
    AIMODataset,
    ASRDataset,
    BurstGPTDataset,
    ConversationDataset,
    HuggingFaceDataset,
    InstructCoderDataset,
    MTBenchDataset,
    NextEditPredictionDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

# 毫秒到秒的转换因子
MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    # 定义一个数据类来存储基准测试的各项指标
    completed: int  # 完成的请求数
    total_input: int  # 总输入 token 数
    total_output: int  # 总输出 token 数
    request_throughput: float  # 请求吞吐量 (req/s)
    request_goodput: float  # 请求“良率”(goodput) (req/s)，即满足特定SLA的请求吞吐量
    output_throughput: float  # 输出吞吐量 (tok/s)
    total_token_throughput: float  # 总 token 吞吐量 (tok/s)
    mean_ttft_ms: float  # 平均首个 token 延迟 (ms)
    median_ttft_ms: float  # 首个 token 延迟的中位数 (ms)
    std_ttft_ms: float  # 首个 token 延迟的标准差 (ms)
    percentiles_ttft_ms: list[tuple[float, float]]  # 首个 token 延迟的百分位数 (ms)
    mean_tpot_ms: float  # 平均每个输出 token 的时间 (ms)
    median_tpot_ms: float  # 每个输出 token 时间的中位数 (ms)
    std_tpot_ms: float  # 每个输出 token 时间的标准差 (ms)
    percentiles_tpot_ms: list[tuple[float, float]]  # 每个输出 token 时间的百分位数 (ms)
    mean_itl_ms: float  # 平均 token 间延迟 (ms)
    median_itl_ms: float  # token 间延迟的中位数 (ms)
    std_itl_ms: float  # token 间延迟的标准差 (ms)
    percentiles_itl_ms: list[tuple[float, float]]  # token 间延迟的百分位数 (ms)
    # E2EL 指的是每个请求的端到端延迟。
    # E2EL stands for end-to-end latency per request.
    # 这是客户端从发送请求到接收到完整响应所花费的时间。
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float  # 平均端到端延迟 (ms)
    median_e2el_ms: float  # 端到端延迟的中位数 (ms)
    std_e2el_ms: float  # 端到端延迟的标准差 (ms)
    percentiles_e2el_ms: list[tuple[float, float]]  # 端到端延迟的百分位数 (ms)


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.
    以指定的速率异步生成请求，可选择带有突发性。

    Args:
        input_requests:
            A list of input requests, each represented as a SampleRequest.
            输入请求列表，每个请求都是一个 SampleRequest。
        request_rate:
            The rate at which requests are generated (requests/s).
            生成请求的速率 (请求数/秒)。
        burstiness (optional):
            The burstiness factor of the request generation.
            请求生成的突发性因子。
            Only takes effect when request_rate is not inf.
            仅当 request_rate 不是无穷大时生效。
            Default value is 1, which follows a Poisson process.
            默认值为 1，遵循泊松过程。
            Otherwise, the request intervals follow a gamma distribution.
            否则，请求间隔遵循伽马分布。
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
            较低的突发性值 (0 < burstiness < 1) 会导致更具突发性的请求，
            而较高的突发性值 (burstiness > 1) 会导致更均匀的请求到达。
    """
    input_requests: Iterable[SampleRequest] = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    # 计算尺度参数 theta 以维持期望的 request_rate。
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
        f"期望突发性因子为正数，但给定的值为 {burstiness}。"
    )
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            # 如果请求速率是无穷大，则无需等待。
            continue

        # Sample the request interval from the gamma distribution.
        # 从伽马分布中采样请求间隔。
        # If burstiness is 1, it follows exponential distribution.
        # 如果突发性为 1，则遵循指数分布。
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        # 下一个请求将在此间隔后发送。
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    """计算基准测试的各项性能指标。"""
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                # 对于某些服务后端，我们使用分词器来计算输出 token 的数量，
                # 而不是查看 len(outputs[i].itl)，因为多个输出 token 可能会被打包在一起。
                # 注意：这可能会稍微增加输出 token 的数量。
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                # 计算每个输出 token 的时间 (TPOT)
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            # 注意：如果 output_len <= 1，我们在计算 goodput 时将 tpot 视为 0
            all_tpots.append(tpot)
            itls += outputs[i].itl  # 收集 token 间延迟
            ttfts.append(outputs[i].ttft)  # 收集首个 token 延迟
            e2els.append(outputs[i].latency)  # 收集端到端延迟
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        # 如果配置了 goodput（良率）计算
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        # 遍历每个请求的指标，判断是否满足所有SLA（服务水平目标）
        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
        # "所有请求都失败了。这很可能是由于基准测试参数配置错误造成的。"

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
                 # 如果后端不支持流式传输，ttfts 列表可能为空
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]],
    extra_body: Optional[dict],
):
    """主基准测试函数。"""
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")  # 未知的后端

    print("Starting initial single prompt test run...") # 开始初始的单个提示测试运行...
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multi_modal_data,
    )

    assert test_mm_content is None or isinstance(test_mm_content, dict)
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )

    # 发送一个测试请求，确保服务正常
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
            # "初始测试运行失败 - 请确保基准测试参数已正确指定。错误: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")
        # "初始测试运行完成。开始主基准测试运行..."

    if lora_modules:
        # For each input request, choose a LoRA module at random.
        # 对每个输入请求，随机选择一个 LoRA 模块。
        lora_modules = iter(
            [random.choice(lora_modules) for _ in range(len(input_requests))]
        )

    if profile:
        # 如果启用了性能分析
        print("Starting profiler...") # 启动分析器...
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started") # 分析器已启动

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
    # 根据 burstiness 值确定请求分布模型

    print(f"Traffic request rate: {request_rate}") # 流量请求速率
    print(f"Burstiness factor: {burstiness} ({distribution})") # 突发性因子
    print(f"Maximum request concurrency: {max_concurrency}") # 最大请求并发数

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    # 使用信号量来控制最大并发数
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        """一个包装函数，用于在发送请求前获取信号量，以限制并发。"""
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    # 异步生成并发送所有请求
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
        )
        req_model_id, req_model_name = model_id, model_name
        if lora_modules:
            req_lora_module = next(lora_modules)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        # 如果启用了性能分析，则停止分析器
        print("Stopping profiler...") # 停止分析器...
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped") # 分析器已停止

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # 计算并整理性能指标
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    # 打印结果
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed)) # 成功请求数
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration)) # 基准测试持续时间(秒)
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input)) # 总输入 token 数
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output)) # 总生成 token 数
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        ) # 请求吞吐量 (req/s)
    )
    if goodput_config_dict:
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            ) # 请求良率 (req/s)
        )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        ) # 输出 token 吞吐量 (tok/s)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        ) # 总 token 吞吐量 (tok/s)
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        # 此函数打印并添加指定指标的统计信息。
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    # 检查并解析 goodput（良率）参数
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                    # f"找到无效的指标名称, {slo_name}: {slo_val}。"
                    # "服务水平目标名称应为 "
                    # f"{str(VALID_NAMES)} 中的一个。"
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                    # f"找到无效的值, {slo_name}: {slo_val}。"
                    # "服务水平目标值应为非负数。"
                )
    return goodput_config_dict


def parse_goodput(slo_pairs):
    """解析 goodput 的键值对参数。"""
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
            # "服务水平目标格式无效。"
            # '请以 "键:值" 对的形式指定 goodput 的服务水平目标，'
            # "其中键是指标名称，值是毫秒数。"
        ) from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    """将结果保存为 PyTorch 基准测试格式。"""
    metrics = [
        "median_ttft_ms",
        "mean_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    # 这些原始数据可能有用，但它们相当大。如果需要，可以稍后添加。
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        # 这里不要使用 .json 后缀，因为我们不希望 CI 把它捡起来
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    """程序主入口函数。"""
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
            # "请指定 '--dataset-name' 以及（如果需要）相应的 '--dataset-path'。"
        )

    # 根据数据集名称加载和采样数据
    if args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        # 对于 "sonnet" 数据集，格式化取决于后端。
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
                # "分词器/模型必须有聊天模板才能使用 sonnet 数据集。"
            )
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        # 以下所有数据集都从 HuggingFaceDataset 基类实现
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ConversationDataset
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif args.dataset_path in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS:  # noqa: E501
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ASRDataset
            args.hf_split = "train"
        else:
            supported_datasets = set(
                [
                    dataset_name
                    for cls in HuggingFaceDataset.__subclasses__()
                    for dataset_name in cls.SUPPORTED_DATASET_PATHS
                ]
            )
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats."
                # f"不支持的数据集路径: {args.dataset_path}。"
                # "Huggingface 数据集仅支持来自以下之一的 dataset_path："
                # f"{supported_datasets}。"
                # "如果您想添加对其他数据集格式的支持，请考虑贡献。"
            )

        if dataset_class.IS_MULTIMODAL and backend not in [
            "openai-chat",
            "openai-audio",
        ]:
            # multi-modal benchmark is only available on OpenAI Chat backend.
            # 多模态基准测试仅在 OpenAI Chat 后端上可用。
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' backend."
                # "多模态内容仅在 'openai-chat' 和 'openai-audio' 后端上受支持。"
            )
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        # 对于遵循相似结构的数据集，使用一个映射。
        dataset_mapping = {
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.sharegpt_output_len,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(tokenizer=tokenizer, num_requests=args.num_prompts),
            "random": lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
            ),
        }

        try:
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err
            # 未知的数据集
    goodput_config_dict = check_goodput_args(args)

    # Collect the sampling parameters.
    # 收集采样参数。
    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
        }.items()
        if v is not None
    }

    # Sampling parameters are only supported by openai-compatible backend.
    # 采样参数仅受 openai-compatible 后端支持。
    if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError(
            "Sampling parameters are only supported by openai-compatible backends."
            # "采样参数仅在与 openai 兼容的后端上受支持。"
        )

    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0  # Default to greedy decoding.
                                             # 默认为贪心解码。

    # Avoid GC processing "static" data - reduce pause times.
    # 避免垃圾回收处理“静态”数据 - 减少暂停时间。
    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            lora_modules=args.lora_modules,
            extra_body=sampling_params,
        )
    )

    # Save config and results to json
    # 将配置和结果保存到 json 文件
    if args.save_result or args.append_result:
        result_json: dict[str, Any] = {}

        # Setup
        # 设置
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        # 元数据
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                        # "元数据格式无效。请使用 键=值 格式。"
                    )
        # Traffic
        # 流量
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        # 与基准测试结果合并
        result_json = {**result_json, **benchmark_result}

        if not args.save_detailed:
            # Remove fields with too many data points
            # 删除数据点过多的字段
            for field in [
                "input_lens",
                "output_lens",
                "ttfts",
                "itls",
                "generated_texts",
                "errors",
            ]:
                if field in result_json:
                    del result_json[field]

        # Save to file
        # 保存到文件
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            # Append a newline.
            # 追加一个换行符。
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
                    "基准测试在线服务吞吐量。"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="后端类型。",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="服务器或 API 的基础 URL，如果不使用 http 主机和端口，则使用此参数。",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    # 这里使用 127.0.0.1 而不是 localhost 来强制使用 ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址。")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口号。")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API 端点。",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf"],
        help="用于基准测试的数据集名称。",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="sharegpt/sonnet 数据集的路径。或者，如果使用 HF 数据集，则为 huggingface 数据集 ID。",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="最大并发请求数。这可以用来模拟上层组件强制执行最大并发请求数环境。"
             "虽然 --request-rate 参数控制请求发起的速率，但此参数将控制实际允许同时执行的请求数量。"
             "这意味着，当两者结合使用时，如果服务器处理请求的速度不够快，"
             "实际请求速率可能会低于 --request-rate 指定的速率。",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称。",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="分词器的名称或路径，如果不使用默认分词器。",  # noqa: E501
    )
    parser.add_argument("--use-beam-search", action="store_true", help="是否使用束搜索。")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="要处理的提示数量。",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="每个 token 要计算并作为请求一部分返回的 logprobs 数量。"
             "如果未指定，则 (1) 如果禁用束搜索，则不计算 logprobs 并为每个 token 返回一个虚拟 logprob；"
             "或 (2) 如果启用束搜索，则为每个 token 计算 1 个 logprob。",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="每秒请求数。如果为 inf，则所有请求都在时间 0 发送。"
             "否则，我们使用泊松过程或伽马分布来合成请求到达时间。",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="请求生成的突发性因子。"
             "仅在 request_rate 不为 inf 时生效。"
             "默认值为 1，遵循泊松过程。"
             "否则，请求间隔遵循伽马分布。"
             "较低的突发性值 (0 < burstiness < 1) 会导致更具突发性的请求。"
             "较高的突发性值 (burstiness > 1) 会导致更均匀的请求到达。",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="信任来自 huggingface 的远程代码。",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="指定以禁用 tqdm 进度条。",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="使用 Torch Profiler。端点必须使用 VLLM_TORCH_PROFILER_DIR 启动以启用分析器。",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="指定将基准测试结果保存到 json 文件。",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="保存结果时，是否包括每个请求的信息，如响应、错误、ttfs、tpots 等。",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="将基准测试结果追加到现有的 json 文件。",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="键值对（例如，--metadata version=0.3.3 tp=1），"
             "用于本次运行的元数据，将被保存在结果 JSON 文件中以供记录。",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="指定保存基准测试 json 结果的目录。如果未指定，结果将保存在当前目录中。",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="指定保存基准测试 json 结果的文件名。"
             "如果未指定，结果将以 "
             "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
             " 格式保存。",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="发送基准测试请求时设置 ignore_eos 标志。"
             "警告：deepspeed_mii 和 tgi 不支持 ignore_eos。",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="以逗号分隔的所选指标列表，用于报告百分位数。"
             "此参数指定要报告百分位数的指标。"
             '允许的指标名称为 "ttft", "tpot", "itl", "e2el"。'
             '默认值为 "ttft,tpot,itl"。',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="所选指标的百分位数列表，以逗号分隔。"
             '要报告第 25、50 和 75 百分位数，请使用 "25,50,75"。'
             '默认值为 "99"。'
             '使用 "--percentile-metrics" 选择指标。',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='以 "键:值" 对的形式为“良率”(goodput)指定服务水平目标，'
             "其中键是指标名称，值是毫秒数。"
             '可以提供多个 "键:值" 对，用空格分隔。'
             '允许的请求级别指标名称为 "ttft", "tpot", "e2el"。'
             "有关 goodput 定义的更多背景信息，请参阅 DistServe 论文: https://arxiv.org/pdf/2401.09670 "
             "和博客: https://hao-ai-lab.github.io/blogs/distserve",
    )

    # group for dataset specific arguments
    # 数据集特定参数组
    sonnet_group = parser.add_argument_group("sonnet dataset options", "sonnet 数据集选项")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="每个请求的输入 token 数，仅用于 sonnet 数据集。",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="每个请求的输出 token 数，仅用于 sonnet 数据集。",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="每个请求的前缀 token 数，仅用于 sonnet 数据集。",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options", "sharegpt 数据集选项")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="每个请求的输出长度。覆盖 ShareGPT 数据集中的输出长度。",
    )

    random_group = parser.add_argument_group("random dataset options", "随机数据集选项")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="每个请求的输入 token 数，仅用于随机采样。",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="每个请求的输出 token 数，仅用于随机采样。",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="用于采样输入/输出长度的范围比率，仅用于随机采样。"
             "必须在 [0, 1) 范围内，以定义一个对称的采样范围"
             "[length * (1 - range_ratio), length * (1 + range_ratio)]。",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="请求中随机上下文之前的固定前缀 token 数量。"
             "总输入长度是 `random-prefix-len` 和从 "
             "[input_len * (1 - range_ratio), input_len * (1 + range_ratio)] "
             "中采样的随机上下文长度之和。",
    )

    hf_group = parser.add_argument_group("hf dataset options", "HF 数据集选项")
    hf_group.add_argument(
        "--hf-subset", type=str, default=None, help="HF 数据集的子集。"
    )
    hf_group.add_argument(
        "--hf-split", type=str, default=None, help="HF 数据集的划分。"
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="每个请求的输出长度。覆盖从采样的 HF 数据集中的输出长度。",
    )

    sampling_group = parser.add_argument_group("sampling parameters", "采样参数")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p 采样参数。仅对与 openai 兼容的后端有效。",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k 采样参数。仅对与 openai 兼容的后端有效。",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p 采样参数。仅对与 openai 兼容的后端有效。",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="温度采样参数。仅对与 openai 兼容的后端有效。"
             "如果未指定，则默认为贪心解码 (即 temperature==0.0)。",
    )

    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral", "custom"],
        help='分词器模式。\n\n* "auto" 将使用可用的快速分词器。\n* "slow" 将'
             "始终使用慢速分词器。\n* "
             '"mistral" 将始终使用 `mistral_common` 分词器。\n*'
             '"custom" 将使用 --tokenizer 选择预注册的分词器。',
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="API 中使用的模型名称。"
             "如果未指定，模型名称将与 "
             "``--model`` 参数相同。",
    )

    parser.add_argument(
        "--lora-modules",
        nargs="+",
        default=None,
        help="启动服务器时传入的 LoRA 模块名称的子集。"
             "对于每个请求，脚本会随机选择一个 LoRA 模块。",
    )

    args = parser.parse_args()

    main(args)
