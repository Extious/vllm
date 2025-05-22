import os
import json
from typing import List, Optional, Tuple, Dict

# 设置环境变量
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"  # 优先使用xformers的memory_efficient_attention_forward函数

# 导入必要的库
import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Initialize the large model
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95,max_model_len=10000)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm.set_tokenizer(tokenizer)

class PromptFieldTracker:
    """Simplified tracker to construct full prompt and input_ids."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = None  # Stores the final input_ids for the full prompt

    def track_positions(self, prefix_prompt, track_data, query_prompt):
        """
        Constructs the full prompt and corresponding input_ids.

        Args:
            prefix_prompt: Prompt for prefix like system prompt.
            track_data: List of item data (dictionaries) to be included in the prompt.
            query_prompt: Query prompt.

        Returns:
            Tuple of (final_input_ids for the full prompt, list of token_id_chunks).
            The list of chunks is returned for signature compatibility but may be unused by the caller.
        """
        full_prompt_str = prefix_prompt

        # Encode prefix, removing BOS token as per original logic
        prefix_token_ids = self.tokenizer.encode(prefix_prompt)[1:]

        item_token_ids_chunks = [] # To store token IDs of individual item prompts
        for item_data in track_data:
            item_prefix_str = f"\n- "  # Formatting for each item
            item_json_str = json.dumps(item_data) # Convert item dict to JSON string
            item_prompt_segment_str = item_prefix_str + item_json_str

            # Encode item segment, removing BOS token
            current_item_tokens = self.tokenizer.encode(item_prompt_segment_str)[1:]
            item_token_ids_chunks.append(current_item_tokens)

            full_prompt_str += item_prompt_segment_str # Append current item string to the full prompt string

        full_prompt_str += query_prompt # Append the final query string

        # Encode query, removing BOS token
        query_token_ids = self.tokenizer.encode(query_prompt)[1:]

        # Generate the final input_ids from the fully constructed prompt string, removing BOS token
        self.input_ids = self.tokenizer.encode(full_prompt_str)[1:]

        # Construct all_chunk_ids (list of token ID lists, each without BOS)
        # This is kept for signature compatibility with the original call site.
        all_chunk_ids = [prefix_token_ids] + item_token_ids_chunks + [query_token_ids]

        return self.input_ids, all_chunk_ids

def load_user_data(user_id):
    """Load user historical purchase data and candidates"""
    user_dir = os.path.join(os.path.dirname(__file__),"..", "dataset", user_id)

    # Read historical purchase records
    history_path = os.path.join(user_dir, "history.json")
    with open(history_path, 'r') as f:
        history_data = json.load(f)

    # Read candidates
    candidate_path = os.path.join(user_dir, "candidate.json")
    with open(candidate_path, 'r') as f:
        candidate_data = json.load(f)

    return history_data, candidate_data

def visualize_grouped_attention_density(
    attn_weights: torch.Tensor,
    tokens: List[str],
    output_path: str = "grouped_attention_density.png",
    head_idx: Optional[int] = None,
    normalize: bool = True,
    focus_range: Optional[Tuple[int, int]] = None,
    highlight_tokens: Optional[List[int]] = None,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    cmap: str = 'YlOrRd',
    group_size: int = 50,
) -> Dict[str, str]:
    """可视化分组后的注意力权重矩阵，适用于长序列

    Args:
        attn_weights: 注意力权重张量，形状为 [num_heads, seq_len, seq_len]
        tokens: 输入序列的token列表
        output_path: 输出图像的基础路径
        head_idx: 要可视化的注意力头索引，None表示使用第一个头（head 0）
        normalize: 是否对注意力权重进行归一化
        focus_range: 可选的关注范围，格式为(start_idx, end_idx)，只可视化这个范围内的token
        highlight_tokens: 可选的高亮token索引列表，这些token将在可视化中被特别标记
        plot_title: 可选的图表标题，如果为None则使用默认标题
        figsize: 图表大小
        dpi: 图表分辨率
        cmap: 颜色映射，默认为'YlOrRd'，注意力分数越大颜色越深
        group_size: 每组包含的token数量，默认为50

    Returns:
        包含生成的图像路径的字典
    """
    # 导入matplotlib（仅在需要时导入，避免不必要的依赖）
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np

    # 确保输入数据在CPU上并转换为numpy数组
    if attn_weights.device != torch.device('cpu'):
        attn_weights = attn_weights.cpu()

    # 处理注意力权重
    if head_idx is not None:
        # 使用特定的注意力头
        if head_idx >= attn_weights.shape[0]:
            raise ValueError(f"head_idx {head_idx} 超出范围，最大值为 {attn_weights.shape[0]-1}")
        # 确保转换为float32类型，避免BFloat16不兼容问题
        attn = attn_weights[head_idx].to(torch.float32).numpy()
        head_info = f"head {head_idx}"
    else:
        # 使用第一个头作为默认值
        attn = attn_weights[0].to(torch.float32).numpy()
        head_info = "head 0"

    # 应用focus_range（如果提供）
    if focus_range is not None:
        start_idx, end_idx = focus_range
        if start_idx < 0 or end_idx > len(tokens) or start_idx >= end_idx:
            raise ValueError(f"无效的focus_range: {focus_range}")
        attn = attn[start_idx:end_idx, start_idx:end_idx]
        tokens = tokens[start_idx:end_idx]

    seq_len = attn.shape[0]

    # 确保tokens长度与注意力矩阵匹配
    if len(tokens) != seq_len:
        print(f"警告: tokens长度({len(tokens)})与注意力矩阵的序列长度({seq_len})不匹配")
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        else:
            tokens = tokens + [""] * (seq_len - len(tokens))

    # 计算分组数量
    num_groups = (seq_len + group_size - 1) // group_size  # 向上取整

    # 创建分组后的注意力矩阵和token列表
    grouped_attn = np.zeros((num_groups, num_groups))
    grouped_tokens = []

    # 对tokens进行分组，每组取第一个token作为代表
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, seq_len)
        # 使用组内第一个token作为该组的代表
        group_token = f"Group {i+1}: {tokens[start_idx]}"
        grouped_tokens.append(group_token)

        # 计算分组后的注意力权重（每组内的平均值）
        for j in range(num_groups):
            j_start = j * group_size
            j_end = min((j + 1) * group_size, seq_len)
            # 计算两组之间的平均注意力权重
            grouped_attn[i, j] = np.mean(attn[start_idx:end_idx, j_start:j_end])

    # 创建输出路径字典
    output_paths = {}

    # 生成密度图
    plt.figure(figsize=figsize)

    # 使用热力图直接可视化分组后的注意力权重矩阵，使用YlOrRd颜色映射使注意力分数越大颜色越深
    heatmap = plt.imshow(grouped_attn, cmap=cmap, aspect='auto')
    plt.colorbar(heatmap, label='Average attention weight')
    plt.xlabel('Key position group (tokens being attended to)')
    plt.ylabel('Query position group (current tokens)')

    # 设置标题
    title = plot_title if plot_title else f'Grouped attention weights ({head_info}, group size={group_size})'
    plt.title(title)

    # 添加分组token标签
    plt.xticks(range(num_groups), grouped_tokens, rotation=45, ha='right')
    plt.yticks(range(num_groups), grouped_tokens)

    # 高亮特定token组（如果提供）
    if highlight_tokens:
        for idx in highlight_tokens:
            group_idx = idx // group_size
            if 0 <= group_idx < num_groups:
                plt.axhline(y=group_idx, color='red', linestyle='--', alpha=0.5)
                plt.axvline(x=group_idx, color='red', linestyle='--', alpha=0.5)

    # 保存图像
    plt.tight_layout()
    density_path = output_path
    plt.savefig(density_path, dpi=dpi)
    plt.close()

    output_paths['density'] = density_path
    print(f"分组注意力权重可视化已保存到 {density_path}")

    return output_paths

def visualize_attention_density(
    attn_weights: torch.Tensor,
    tokens: List[str],
    output_path: str = "attention_density.png",
    head_idx: Optional[int] = None,
    normalize: bool = True,
    focus_range: Optional[Tuple[int, int]] = None,
    highlight_tokens: Optional[List[int]] = None,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    cmap: str = 'YlOrRd',
    token_batch_size: int = 30,
) -> Dict[str, str]:
    """可视化注意力权重矩阵

    Args:
        attn_weights: 注意力权重张量，形状为 [num_heads, seq_len, seq_len]
        tokens: 输入序列的token列表
        output_path: 输出图像的基础路径
        head_idx: 要可视化的注意力头索引，None表示使用第一个头（head 0）
        normalize: 是否对注意力权重进行归一化
        focus_range: 可选的关注范围，格式为(start_idx, end_idx)，只可视化这个范围内的token
        highlight_tokens: 可选的高亮token索引列表，这些token将在可视化中被特别标记
        plot_title: 可选的图表标题，如果为None则使用默认标题
        figsize: 图表大小
        dpi: 图表分辨率
        cmap: 颜色映射，默认为'YlOrRd'，注意力分数越大颜色越深
        token_batch_size: 当序列太长时，每个批次显示的token数量

    Returns:
        包含生成的图像路径的字典
    """
    # 导入matplotlib（仅在需要时导入，避免不必要的依赖）
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    # 确保输入数据在CPU上并转换为numpy数组
    if attn_weights.device != torch.device('cpu'):
        attn_weights = attn_weights.cpu()

    # 处理注意力权重
    if head_idx is not None:
        # 使用特定的注意力头
        if head_idx >= attn_weights.shape[0]:
            raise ValueError(f"head_idx {head_idx} 超出范围，最大值为 {attn_weights.shape[0]-1}")
        # 确保转换为float32类型，避免BFloat16不兼容问题
        attn = attn_weights[head_idx].to(torch.float32).numpy()
        head_info = f"head {head_idx}"
    else:
        # 不再支持平均所有注意力头，而是使用第一个头作为默认值
        attn = attn_weights[0].to(torch.float32).numpy()
        head_info = "head 0"

    # 应用focus_range（如果提供）
    if focus_range is not None:
        start_idx, end_idx = focus_range
        if start_idx < 0 or end_idx > len(tokens) or start_idx >= end_idx:
            raise ValueError(f"无效的focus_range: {focus_range}")

        # 保存原始token列表，用于在可视化时显示实际token值
        original_tokens = tokens.copy()

        # 截取focus_range范围内的注意力权重和token
        attn = attn[start_idx:end_idx, start_idx:end_idx]
        tokens = tokens[start_idx:end_idx]

        # 打印focus范围内的token信息，便于调试
        print(f"可视化focus范围: {start_idx} 到 {end_idx}，共 {end_idx - start_idx} 个token")
        if end_idx - start_idx <= 10:  # 只打印少量token作为示例
            for i, token in enumerate(tokens):
                print(f"Token {start_idx + i}: {token}")

    seq_len = attn.shape[0]

    # 确保tokens长度与注意力矩阵匹配
    if len(tokens) != seq_len:
        print(f"警告: tokens长度({len(tokens)})与注意力矩阵的序列长度({seq_len})不匹配")
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        else:
            tokens = tokens + [""] * (seq_len - len(tokens))

    # 创建输出路径字典
    output_paths = {}

    # 生成密度图
    # 创建图形
    plt.figure(figsize=figsize)

    # 使用热力图直接可视化注意力权重矩阵，使用YlOrRd颜色映射使注意力分数越大颜色越深
    heatmap = plt.imshow(attn, cmap=cmap, aspect='auto')
    plt.colorbar(heatmap, label='Attention weight')
    plt.xlabel('Key position (token being attended to)')
    plt.ylabel('Query position (current token)')

    # 设置标题
    title = plot_title if plot_title else f'Attention weights ({head_info})'
    plt.title(title)

    # 添加token标签
    if seq_len <= token_batch_size:
        # 对于短序列，显示所有token
        # 为每个位置创建标签，显示实际token值和位置
        token_labels = [f"{i}: {token}" for i, token in enumerate(tokens)]
        plt.xticks(range(seq_len), token_labels, rotation=45, ha='right', fontsize=8)
        plt.yticks(range(seq_len), token_labels, fontsize=8)
    elif focus_range is not None:
        # 对于focus_range内的token（如最后50个token），显示所有实际token值
        start_idx, end_idx = focus_range
        focus_len = end_idx - start_idx

        # 如果focus范围内的token数量较少（如最后50个token），显示所有token的实际值
        if focus_len <= token_batch_size:
            # 创建位置索引列表（对应于注意力矩阵中的位置）
            positions = list(range(seq_len))

            # 为每个位置创建标签，显示实际token值和原始位置
            # 注意：这里的tokens已经是focus_range截取后的结果
            token_labels = []
            for i in range(seq_len):
                # 计算原始位置（在focus_range截取前的位置）
                original_pos = i + start_idx
                # 为每个token添加原始位置信息，使其更易于识别
                token_labels.append(f"{original_pos}: {tokens[i]}")

            # 设置x轴和y轴的刻度和标签
            plt.xticks(positions, token_labels, rotation=45, ha='right', fontsize=8)
            plt.yticks(positions, token_labels, fontsize=8)
        else:
            # 对于较长的focus范围，只显示部分标签，但仍然包含位置信息
            step = max(1, seq_len // 20)
            x_positions = list(range(0, seq_len, step))
            x_labels = []
            for i in x_positions:
                # 计算原始位置（在focus_range截取前的位置）
                original_pos = i + start_idx
                x_labels.append(f"{original_pos}: {tokens[i]}")
            plt.xticks(x_positions, x_labels, rotation=45, ha='right', fontsize=8)
            plt.yticks(x_positions, x_labels, fontsize=8)
    else:
        # 对于长序列，只显示部分标签，但添加位置信息
        step = max(1, seq_len // 20)
        x_positions = list(range(0, seq_len, step))
        x_labels = [f"{i}: {tokens[i]}" for i in x_positions]
        plt.xticks(x_positions, x_labels, rotation=45, ha='right', fontsize=8)
        plt.yticks(x_positions, x_labels, fontsize=8)

    # 高亮特定token（如果提供）
    if highlight_tokens:
        for idx in highlight_tokens:
            if 0 <= idx < seq_len:
                plt.axhline(y=idx, color='red', linestyle='--', alpha=0.5)
                plt.axvline(x=idx, color='red', linestyle='--', alpha=0.5)

    # 保存图像
    plt.tight_layout()
    density_path = output_path
    plt.savefig(density_path, dpi=dpi)
    plt.close()

    output_paths['density'] = density_path
    print(f"注意力权重可视化已保存到 {density_path}")

    return output_paths

def generate_recommendation_with_cacheblend(user_id):
    """
    为指定用户生成推荐并可视化注意力权重

    Args:
        user_id: 用户ID

    Returns:
        生成的输出结果
    """
    # Load user data
    history_data, candidate_data = load_user_data(user_id)
    all_items = history_data + candidate_data

    # Extract username (from user ID)
    username = user_id.split('_')[-1]

    # Construct the basic prompt prefix
    prefix_prompt = f"You are an intelligent assistant that can rank items based on the user's preference.\nAnalyze the provided purchase history and candidate items to identify user preferences and purchase patterns. Then, rank the candidate items based on their alignment with the user's preferences and other contextual factors. All the items should be included and listed using identifiers, in descending order of the user's preference.\n"

    # Create the query prompt
    query_prompt = f"""\n\n All items related are above. The first {len(history_data)} items are the history items {username} has purchased. The rest {len(candidate_data)} items are candidates.\n
    Please rank the candidates. The most preferred recommendation item should be listed first. The output format should be [] > [], where each [] is an identifier, e.g., [1] > [2]. Only respond with the {len(candidate_data)} JSON format ranking results, do not say any word or explain. Output in the following JSON format:
{{\"rank\": \"[] > [] .. > []\"}} Do not say any word except this JSON format."""

    # Create an instance of the simplified PromptFieldTracker
    tracker = PromptFieldTracker(tokenizer)

    print(f"加载的历史记录数: {len(history_data)}")
    print(f"加载的候选项数: {len(candidate_data)}")

    # 获取完整提示词的token IDs
    input_ids_from_tracker, _ = tracker.track_positions(prefix_prompt, all_items, query_prompt)

    # 将token IDs解码为完整提示词字符串
    input_prompt = tokenizer.decode(input_ids_from_tracker)

    # 获取cache_metadata引用
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata

    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)

    # 设置缓存元数据
    cache_metadata["check"] = False
    cache_metadata['collect'] = False

    # 启用注意力权重返回 - 这是获取真实注意力权重的关键步骤
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.return_attn_weights = True

    # 生成输出
    output = llm.generate([input_prompt], sampling_params)

    # 获取注意力权重
    attn_weights = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.first_layer_attn_weights

    # 如果未能获取到注意力权重，尝试从模型的其他位置获取
    if attn_weights is None:
        print("尝试从模型的其他位置获取注意力权重...")
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn'):
                if hasattr(layer.self_attn.attn, 'last_attn_weights'):
                    attn_weights = layer.self_attn.attn.last_attn_weights
                    print(f"从第{i}层获取到注意力权重")
                    break

    # 如果成功获取到注意力权重，则进行可视化
    if attn_weights is not None:
        print("成功获取注意力权重，正在生成可视化...")
        # 获取输入的token列表
        tokens = tokenizer.convert_ids_to_tokens(input_ids_from_tracker)
        # 将注意力权重转换为float32类型，以避免BFloat16不兼容问题
        attn_weights = attn_weights.to(torch.float32)

        print(f"注意力权重形状: {attn_weights.shape}")

        # 确保注意力权重的形状正确 [num_heads, seq_len, seq_len]
        if len(attn_weights.shape) == 3:
            _, seq_len_q, _ = attn_weights.shape  # 只需要使用seq_len_q来检查token列表长度

            # 检查token列表长度与注意力权重的序列长度是否匹配
            if len(tokens) != seq_len_q:
                # 如果不匹配，可能需要截断或填充token列表
                if len(tokens) > seq_len_q:
                    tokens = tokens[:seq_len_q]
                    print(f"已截断tokens列表至{len(tokens)}个token")
                else:
                    # 如果token列表太短，填充空字符串
                    tokens = tokens + [""] * (seq_len_q - len(tokens))
                    print(f"已填充tokens列表至{len(tokens)}个token")

            # 获取序列长度
            seq_len = attn_weights.shape[1]
            print(f"序列长度: {seq_len}")

            # 使用真实的注意力权重数据进行可视化
            # 计算最后50个token的注意力
            focus_range = (max(0, seq_len - 50), seq_len)
            output_paths = visualize_attention_density(
                attn_weights=attn_weights,
                tokens=tokens,
                output_path=f"attention_density_{user_id}_head_5.png",
                head_idx=5,  # 使用第5个头
                normalize=True,
                focus_range=focus_range,
                highlight_tokens=None,  # 可以根据需要高亮特定token
                plot_title=f"{user_id} head 5 attention scores (last 50 tokens)",
                token_batch_size=50  # 确保最后50个token都能显示
            )

            # 如果序列长度超过500，使用分组注意力可视化
            if seq_len > 500:
                print(f"序列长度({seq_len})较长，使用分组注意力可视化...")
                grouped_output_paths = visualize_grouped_attention_density(
                    attn_weights=attn_weights,
                    tokens=tokens,
                    output_path=f"grouped_attention_density_{user_id}_head_15.png",
                    head_idx=15,  # 使用第5个头
                    normalize=True,
                    group_size=50,  # 每50个token分为一组
                    plot_title=f"{user_id} head 15 grouped attention scores (group size=50)"
                )
                # 合并输出路径
                output_paths.update(grouped_output_paths)
        else:
            print(f"错误: 注意力权重形状不正确: {attn_weights.shape}，应为 [num_heads, seq_len, seq_len]")
    else:
        print("错误: 未能获取注意力权重，无法生成可视化")

    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")

    return output


# def analyze_attention_patterns(
#     attn_weights: torch.Tensor,
#     tokens: List[str],
#     user_id: str
# ) -> None:
#     """
#     分析注意力模式并生成多种可视化

#     Args:
#         attn_weights: 注意力权重张量
#         tokens: token列表
#         user_id: 用户ID
#     """
#     num_heads = attn_weights.shape[0]
#     seq_len = attn_weights.shape[1]

#     print(f"分析注意力模式...")
#     print(f"序列长度: {seq_len}, 注意力头数量: {num_heads}")

#     # 如果序列很长，可以只关注最后部分token
#     if seq_len > 100:
#         # 关注最后50个token
#         focus_range = (max(0, seq_len - 50), seq_len)
#         visualize_attention_density(
#             attn_weights=attn_weights,
#             tokens=tokens,
#             output_path=f"attention_last50_{user_id}.png",
#             head_idx=0,  # 使用第一个头
#             normalize=True,
#             focus_range=focus_range,
#             plot_title=f"{user_id} the last 50 tokens attention analysis"
#         )

#     # 如果序列长度超过500，使用分组注意力可视化
#     if seq_len > 500:
#         print(f"序列长度({seq_len})较长，使用分组注意力可视化...")

#         # 使用不同的分组大小进行可视化
#         for group_size in [50, 100]:
#             visualize_grouped_attention_density(
#                 attn_weights=attn_weights,
#                 tokens=tokens,
#                 output_path=f"grouped_attention_{group_size}_{user_id}.png",
#                 head_idx=0,  # 使用第一个头
#                 normalize=True,
#                 group_size=group_size,  # 每group_size个token分为一组
#                 plot_title=f"{user_id} grouped attention analysis (group size={group_size})"
#             )

#         # 对于特别长的序列，尝试更大的分组大小
#         if seq_len > 1000:
#             visualize_grouped_attention_density(
#                 attn_weights=attn_weights,
#                 tokens=tokens,
#                 output_path=f"grouped_attention_200_{user_id}.png",
#                 head_idx=0,  # 使用第一个头
#                 normalize=True,
#                 group_size=200,  # 每200个token分为一组
#                 plot_title=f"{user_id} grouped attention analysis (group size=200)"
#             )

if __name__ == "__main__":
    # 默认用户ID
    user_id = "user_A1A5YE7K0WHN2T"

    print(f"\n===== 为用户 {user_id} 生成推荐并可视化注意力 =====")

    try:
        # 生成推荐并获取注意力权重可视化
        output = generate_recommendation_with_cacheblend(user_id)
        print(f"推荐生成完成")

        # 获取注意力权重
        attn_weights = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.first_layer_attn_weights

        # 如果成功获取到注意力权重，进行可视化分析
        if attn_weights is not None:
            # 加载用户数据
            history_data, candidate_data = load_user_data(user_id)
            all_items = history_data + candidate_data

            # 构建提示词
            prefix_prompt = f"You are an intelligent assistant that can rank items based on the user's preference.\nAnalyze the provided purchase history and candidate items to identify user preferences and purchase patterns. Then, rank the candidate items based on their alignment with the user's preferences and other contextual factors. All the items should be included and listed using identifiers, in descending order of the user's preference.\n"

            query_prompt = f"""\n\n All items related are above. The first {len(history_data)} items are the history items {user_id.split('_')[-1]} has purchased. The rest {len(candidate_data)} items are candidates.\n
            Please rank the candidates. The most preferred recommendation item should be listed first. The output format should be [] > [], where each [] is an identifier, e.g., [1] > [2]. Only respond with the {len(candidate_data)} JSON format ranking results, do not say any word or explain. Output in the following JSON format:
            {{"rank": "[] > [] .. > []"}} Do not say any word except this JSON format."""

            # 创建PromptFieldTracker实例
            tracker = PromptFieldTracker(tokenizer)

            # 获取token IDs和对应的token
            input_ids = tracker.track_positions(prefix_prompt, all_items, query_prompt)[0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # 将注意力权重转换为float32类型，避免BFloat16不兼容问题
            attn_weights = attn_weights.to(torch.float32)

            # 进行注意力模式分析
            # analyze_attention_patterns(attn_weights, tokens, user_id)

    except Exception as e:
        print(f"为用户 {user_id} 生成推荐时出错：{e}")
        import traceback
        traceback.print_exc()

    print(f"===== 用户 {user_id} 处理结束 =====")
    print("\n===== 测试完成 =====")