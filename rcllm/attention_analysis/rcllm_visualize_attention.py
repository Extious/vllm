import os
import json
from typing import List, Optional, Tuple, Dict

# Set environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"  # Prefer to use xformers' memory_efficient_attention_forward function

# Import necessary libraries
import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd

# Initialize the large model
# llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.8, enforce_eager=True, max_model_len=10000, dtype="half")
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.6, dtype="half")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

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
    """Load user history and candidates from reviewer_data_processed_strict.

    Accepts IDs in either "user_XXXX" or "reviewer_XXXX" format and maps them to
    the dataset filenames. Uses a single reviewer file and splits the first 3
    as history and the rest as candidates.
    """
    base_dir = "/home/comp/24481750/rcllm/amazon/dataset/reviewer_data_processed_strict"

    # Normalize the provided ID to match on-disk filenames
    if user_id.startswith("user_"):
        normalized_id = "reviewer_" + user_id[len("user_"):]
    elif user_id.startswith("reviewer_"):
        normalized_id = user_id
    else:
        # Try both conventions if no prefix is provided
        normalized_id = f"reviewer_{user_id}"

    candidate_paths = [
        os.path.join(base_dir, f"{normalized_id}.json"),
    ]

    # Also try the raw id without any prefix as a fallback
    if normalized_id.startswith("reviewer_"):
        raw_id = normalized_id[len("reviewer_"):]
        candidate_paths.append(os.path.join(base_dir, f"{raw_id}.json"))

    data_file_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            data_file_path = path
            break

    if data_file_path is None:
        raise FileNotFoundError(
            f"Reviewer data file not found. Tried: {', '.join(candidate_paths)}"
        )

    with open(data_file_path, 'r') as f:
        all_data = json.load(f)

    if len(all_data) < 4:
        raise ValueError(
            f"Insufficient data for {user_id}: need at least 4 items (3 history + 1 candidate), got {len(all_data)}"
        )

    # Split data: first 3 items as history, rest as candidates
    history_data = all_data[:3]
    candidate_data = all_data[3:]

    return history_data, candidate_data

def load_copurchase_data() -> pd.DataFrame:
    """Load copurchase data CSV (aligned with latency_collector_star.py).

    The CSV is stored in rcllm/latency_analysis/copurchase_data.csv
    """
    copurchase_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../latency_analysis", "copurchase_data.csv")
    )
    return pd.read_csv(copurchase_file)

def get_copurchase_counts(copurchase_df: pd.DataFrame, candidate_item: dict, history_items: list) -> dict:
    """Get copurchase counts between candidate item and each history item (same logic as latency_collector_star.py)."""
    copurchase_info: Dict[str, int] = {}

    candidate_item_id = candidate_item.get('itemID', '')

    for idx, history_item in enumerate(history_items):
        history_item_id = history_item.get('itemID', '')

        count1 = copurchase_df[
            (copurchase_df['item1'] == candidate_item_id) & (copurchase_df['item2'] == history_item_id)
        ]['copurchase_count'].sum()

        count2 = copurchase_df[
            (copurchase_df['item1'] == history_item_id) & (copurchase_df['item2'] == candidate_item_id)
        ]['copurchase_count'].sum()

        total_count = int(count1 + count2)
        copurchase_info[
            f"Number of users who bought both this item and Item ID {idx+1}"
        ] = total_count

    return copurchase_info

def create_diagram_format_conversation(user_id: str, history_data: list, candidate_data: list, copurchase_df: pd.DataFrame) -> list:
    """Create the diagram-style chat conversation aligned with latency_collector_star.py."""
    system_intro = "You are an intelligent assistant that can rank items based on the user's preference."

    user_history_intro = f"User {user_id} has purchased the following items in this order:"

    history_items = []
    for item in history_data:
        history_items.append(json.dumps(item, indent=2))

    candidate_count = len(candidate_data)
    task_description = (
        f"I will provide you with {candidate_count} items, each indicated by number identifier []. "
        "Analyze the user's purchase history to identify preferences and purchase patterns. "
        "Then, rank the candidate items based on their alignment with the user's preferences and other contextual factors."
    )

    history_content = f"{user_history_intro}\n" + "\n\n".join(history_items) + f"\n\n{task_description}"

    assistant_ack = "Okay, please provide the items."

    candidate_items_blocks = []
    for i, item in enumerate(candidate_data):
        copurchase_info = get_copurchase_counts(copurchase_df, item, history_data)
        item_with_copurchase = item.copy()
        item_with_copurchase.update(copurchase_info)
        candidate_item_text = f"[{i+1}]\n{json.dumps(item_with_copurchase, indent=2)}"
        candidate_items_blocks.append(candidate_item_text)

    assistant_acks = [f"Received item [{i+1}]." for i in range(len(candidate_data))]

    final_instructions = (
        "Analyze the user's purchase history to identify user preferences and purchase patterns.\n"
        f"Then, rank the {candidate_count} items above based on their alignment with the user's preferences and other contextual factors.\n"
        "All the items should be included and listed using identifiers, in descending order of the user's preference.\n"
        "The most preferred recommendation item should be listed first.\n"
        "The output format should be [] > [], where each [] is an identifier, e.g., [1] > [2].\n"
        "Only respond with the ranking results, do not say any word or explain.\n"
        "Output in the following JSON format: { \"rank\": \"[] > [].. > []\" }"
    )

    conversation = [
        {"role": "system", "content": system_intro},
        {"role": "user", "content": history_content},
        {"role": "assistant", "content": assistant_ack},
    ]

    for i in range(len(candidate_data)):
        conversation.append({"role": "user", "content": candidate_items_blocks[i]})
        conversation.append({"role": "assistant", "content": assistant_acks[i]})

    conversation.append({"role": "user", "content": final_instructions})

    return conversation

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
    """Visualize grouped attention weight matrix, suitable for long sequences

    Args:
        attn_weights: Attention weight tensor, shape [num_heads, seq_len, seq_len]
        tokens: Token list of input sequence
        output_path: Base path for output image
        head_idx: Attention head index to visualize, None means use first head (head 0)
        normalize: Whether to normalize attention weights
        focus_range: Optional focus range, format (start_idx, end_idx), only visualize tokens in this range
        highlight_tokens: Optional list of token indices to highlight, these tokens will be specially marked in visualization
        plot_title: Optional plot title, if None use default title
        figsize: Plot size
        dpi: Plot resolution
        cmap: Color mapping, default 'YlOrRd', higher attention scores have darker colors
        group_size: Number of tokens per group, default 50

    Returns:
        Dictionary containing generated image paths
    """
    # Import matplotlib (only import when needed, avoid unnecessary dependencies)
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure input data is on CPU and convert to numpy array
    if attn_weights.device != torch.device('cpu'):
        attn_weights = attn_weights.cpu()

    # Process attention weights
    if head_idx is not None:
        # Use specific attention head
        if head_idx >= attn_weights.shape[0]:
            raise ValueError(f"head_idx {head_idx} exceeds range, maximum value is {attn_weights.shape[0]-1}")
        # Ensure conversion to float32 type, avoid BFloat16 incompatibility issues
        attn = attn_weights[head_idx].to(torch.float32).numpy()
        head_info = f"head {head_idx}"
    else:
        # Use first head as default
        attn = attn_weights[0].to(torch.float32).numpy()
        head_info = "head 0"

    # Apply focus_range (if provided)
    if focus_range is not None:
        start_idx, end_idx = focus_range
        if start_idx < 0 or end_idx > len(tokens) or start_idx >= end_idx:
            raise ValueError(f"Invalid focus_range: {focus_range}")
        attn = attn[start_idx:end_idx, start_idx:end_idx]
        tokens = tokens[start_idx:end_idx]

    seq_len = attn.shape[0]

    # Ensure tokens length matches attention matrix
    if len(tokens) != seq_len:
        print(f"Warning: tokens length ({len(tokens)}) does not match attention matrix sequence length ({seq_len})")
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        else:
            tokens = tokens + [""] * (seq_len - len(tokens))

    # Calculate number of groups
    num_groups = (seq_len + group_size - 1) // group_size  # Round up

    # Create grouped attention matrix and token list
    grouped_attn = np.zeros((num_groups, num_groups))
    grouped_tokens = []

    # Group tokens, use first token of each group as representative
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, seq_len)
        # Use first token in group as representative for this group
        group_token = f"Group {i+1}: {tokens[start_idx]}"
        grouped_tokens.append(group_token)

        # Calculate grouped attention weights (average within each group)
        for j in range(num_groups):
            j_start = j * group_size
            j_end = min((j + 1) * group_size, seq_len)
            # Calculate average attention weight between two groups
            grouped_attn[i, j] = np.mean(attn[start_idx:end_idx, j_start:j_end])

    # Create output path dictionary
    output_paths = {}

    # Generate density plot
    plt.figure(figsize=figsize)

    # Use heatmap to directly visualize grouped attention weight matrix, use YlOrRd color mapping so higher attention scores have darker colors
    heatmap = plt.imshow(grouped_attn, cmap=cmap, aspect='auto')
    plt.colorbar(heatmap, label='Average attention weight')
    plt.xlabel('Key position group (tokens being attended to)')
    plt.ylabel('Query position group (current tokens)')

    # Set title
    title = plot_title if plot_title else f'Grouped attention weights ({head_info}, group size={group_size})'
    plt.title(title)

    # Add grouped token labels
    plt.xticks(range(num_groups), grouped_tokens, rotation=45, ha='right')
    plt.yticks(range(num_groups), grouped_tokens)

    # Highlight specific token groups (if provided)
    if highlight_tokens:
        for idx in highlight_tokens:
            group_idx = idx // group_size
            if 0 <= group_idx < num_groups:
                plt.axhline(y=group_idx, color='red', linestyle='--', alpha=0.5)
                plt.axvline(x=group_idx, color='red', linestyle='--', alpha=0.5)

    # Save image
    plt.tight_layout()
    density_path = output_path
    plt.savefig(density_path, dpi=dpi)
    plt.close()

    output_paths['density'] = density_path
    print(f"Grouped attention weight visualization saved to {density_path}")

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
    """Visualize attention weight matrix

    Args:
        attn_weights: Attention weight tensor, shape [num_heads, seq_len, seq_len]
        tokens: Token list of input sequence
        output_path: Base path for output image
        head_idx: Attention head index to visualize, None means use first head (head 0)
        normalize: Whether to normalize attention weights
        focus_range: Optional focus range, format (start_idx, end_idx), only visualize tokens in this range
        highlight_tokens: Optional list of token indices to highlight, these tokens will be specially marked in visualization
        plot_title: Optional plot title, if None use default title
        figsize: Plot size
        dpi: Plot resolution
        cmap: Color mapping, default 'YlOrRd', higher attention scores have darker colors
        token_batch_size: Number of tokens to display per batch when sequence is too long

    Returns:
        Dictionary containing generated image paths
    """
    # Import matplotlib (only import when needed, avoid unnecessary dependencies)
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    # Ensure input data is on CPU and convert to numpy array
    if attn_weights.device != torch.device('cpu'):
        attn_weights = attn_weights.cpu()

    # Process attention weights
    if head_idx is not None:
        # Use specific attention head
        if head_idx >= attn_weights.shape[0]:
            raise ValueError(f"head_idx {head_idx} exceeds range, maximum value is {attn_weights.shape[0]-1}")
        # Ensure conversion to float32 type, avoid BFloat16 incompatibility issues
        attn = attn_weights[head_idx].to(torch.float32).numpy()
        head_info = f"head {head_idx}"
    else:
        # No longer support averaging all attention heads, use first head as default
        attn = attn_weights[0].to(torch.float32).numpy()
        head_info = "head 0"

    # Apply focus_range (if provided)
    if focus_range is not None:
        start_idx, end_idx = focus_range

        # 更加详细的focus_range验证
        if start_idx < 0:
            raise ValueError(f"Invalid focus_range: start_idx ({start_idx}) cannot be negative")
        if end_idx > len(tokens):
            raise ValueError(f"Invalid focus_range: end_idx ({end_idx}) exceeds tokens length ({len(tokens)})")
        if start_idx >= end_idx:
            raise ValueError(f"Invalid focus_range: start_idx ({start_idx}) must be less than end_idx ({end_idx})")
        if len(tokens) == 0:
            raise ValueError(f"Invalid focus_range: tokens list is empty, cannot apply focus_range {focus_range}")

        # Save original token list for displaying actual token values in visualization
        original_tokens = tokens.copy()

        # Extract attention weights and tokens within focus_range
        attn = attn[start_idx:end_idx, start_idx:end_idx]
        tokens = tokens[start_idx:end_idx]

        # Print token information within focus range for debugging
        print(f"Visualizing focus range: {start_idx} to {end_idx}, total {end_idx - start_idx} tokens")
        if end_idx - start_idx <= 10:  # Only print a small number of tokens as examples
            for i, token in enumerate(tokens):
                print(f"Token {start_idx + i}: {token}")

    seq_len = attn.shape[0]

    # Ensure tokens length matches attention matrix
    if len(tokens) != seq_len:
        print(f"Warning: tokens length ({len(tokens)}) does not match attention matrix sequence length ({seq_len})")
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        else:
            tokens = tokens + [""] * (seq_len - len(tokens))

    # Create output path dictionary
    output_paths = {}

    # Generate density plot
    # Create figure
    plt.figure(figsize=figsize)

    # Use heatmap to directly visualize attention weight matrix, use YlOrRd color mapping so higher attention scores have darker colors
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
    # Load user data and copurchase data
    history_data, candidate_data = load_user_data(user_id)
    copurchase_df = load_copurchase_data()

    # Build chat-style conversation aligned with latency_collector_star.py
    conversation = create_diagram_format_conversation(user_id, history_data, candidate_data, copurchase_df)

    print(f"Number of loaded history records: {len(history_data)}")
    print(f"Number of loaded candidate items: {len(candidate_data)}")

    # Render conversation to a single prompt string and tokenize for analysis
    input_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    input_ids_from_conversation = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)

    # 获取cache_metadata引用
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata

    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1)

    # 设置缓存元数据
    cache_metadata["check"] = False
    cache_metadata['collect'] = False

    # 启用注意力权重返回 - 这是获取真实注意力权重的关键步骤
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.return_attn_weights = True

    # 生成输出
    output = llm.generate([input_prompt], sampling_params)

    # 获取注意力权重
    attn_weights = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.first_layer_attn_weights

    # 添加调试信息
    print(f"Input prompt length (chars): {len(input_prompt)}")
    print(f"Input token length: {len(input_ids_from_conversation)}")

    # 如果未能获取到注意力权重，尝试从模型的其他位置获取
    if attn_weights is None:
        print("Attempt to obtain attention weights from other positions of the model...")
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn'):
                if hasattr(layer.self_attn.attn, 'last_attn_weights'):
                    attn_weights = layer.self_attn.attn.last_attn_weights
                    print(f"Obtained attention weights from the {i}th layer")
                    break

    # 添加更多调试信息
    if attn_weights is not None:
        print(f"Raw attention weights shape: {attn_weights.shape}")
        print(f"Raw attention weights device: {attn_weights.device}")
        print(f"Raw attention weights dtype: {attn_weights.dtype}")
    else:
        print("Warning: No attention weights obtained from any layer")

    # 如果成功获取到注意力权重，则进行可视化
    if attn_weights is not None:
        print("Successfully obtained attention weights, generating visualization...")
        # 获取输入的token列表
        tokens = tokenizer.convert_ids_to_tokens(input_ids_from_conversation)
        # 将注意力权重转换为float32类型，以避免BFloat16不兼容问题
        attn_weights = attn_weights.to(torch.float32)

        print(f"Attention weight shape:{attn_weights.shape}")

        # 确保注意力权重的形状正确 [num_heads, seq_len, seq_len]
        if len(attn_weights.shape) == 3:
            _, seq_len_q, _ = attn_weights.shape  # 只需要使用seq_len_q来检查token列表长度

            # 检查序列长度是否有效
            if seq_len_q == 0:
                print("Error: Attention weights have zero sequence length, cannot generate visualization")
                return output

            # 检查token列表长度与注意力权重的序列长度是否匹配
            if len(tokens) != seq_len_q:
                # 如果不匹配，可能需要截断或填充token列表
                if len(tokens) > seq_len_q:
                    tokens = tokens[:seq_len_q]
                    print(f"The tokens list has been truncated to {len(tokens)} tokens.")
                else:
                    # 如果token列表太短，填充空字符串
                    tokens = tokens + [""] * (seq_len_q - len(tokens))
                    print(f"The tokens list has been filled to {len(tokens)} tokens.")

            # 获取序列长度
            seq_len = attn_weights.shape[1]
            print(f"Sequence Length:{seq_len}")

            # 检查序列长度是否足够进行可视化
            if seq_len <= 0:
                print("Error: Invalid sequence length for visualization")
                return output

            # 使用真实的注意力权重数据进行可视化
            # 计算最后50个token的注意力，但确保focus_range有效
            if seq_len > 1:
                focus_start = max(0, seq_len - 50)
                focus_end = seq_len
                # 确保focus_range有效（start < end）
                if focus_start < focus_end:
                    focus_range = (focus_start, focus_end)
                else:
                    # 如果序列太短，使用整个序列
                    focus_range = None
            else:
                # 序列长度为1或更少，不使用focus_range
                focus_range = None

            # 只有在focus_range有效或者不使用focus_range时才进行可视化
            if focus_range is None or (focus_range[0] < focus_range[1]):
                output_paths = visualize_attention_density(
                    attn_weights=attn_weights,
                    tokens=tokens,
                    output_path=f"attention_density_{user_id}_head_5.png",
                    head_idx=5,  # 使用第5个头
                    normalize=True,
                    focus_range=focus_range,
                    highlight_tokens=None,  # 可以根据需要高亮特定token
                    plot_title=f"{user_id} head 5 attention scores" + (f" (last {focus_range[1] - focus_range[0]} tokens)" if focus_range else ""),
                    token_batch_size=50  # 确保最后50个token都能显示
                )
            else:
                print(f"Error: Invalid focus_range calculated: {focus_range}, skipping visualization")

            # 如果序列长度超过500，使用分组注意力可视化
            if seq_len > 500:
                print(f"Sequence length ({seq_len}) is longer, using grouped attention visualization...")
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
            print(f"Error: The attention weight shape is incorrect.{attn_weights.shape}, Should be[num_heads, seq_len, seq_len]")
    else:
        print("Error: Unable to obtain attention weights, unable to generate visualization")

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
    user_id = "user_A10FW892S59ABJ"

    print(f"\n=====Generate recommendations and visualize attention for user {user_id}. =====")

    try:
        # 生成推荐并获取注意力权重可视化
        output = generate_recommendation_with_cacheblend(user_id)
        print(f"Recommended generation completed")

        # 获取注意力权重
        attn_weights = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.first_layer_attn_weights

        # 如果成功获取到注意力权重，进行可视化分析
        if attn_weights is not None:
            # 加载用户数据与共购数据，并使用图示风格会话生成token
            history_data, candidate_data = load_user_data(user_id)
            copurchase_df = load_copurchase_data()
            conversation = create_diagram_format_conversation(user_id, history_data, candidate_data, copurchase_df)
            input_ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # 将注意力权重转换为float32类型，避免BFloat16不兼容问题
            attn_weights = attn_weights.to(torch.float32)

            # 进行注意力模式分析
            # analyze_attention_patterns(attn_weights, tokens, user_id)

    except Exception as e:
        print(f"An error occurred while generating recommendations for user {user_id}: {e}")
        import traceback
        traceback.print_exc()

    print(f"=====User {user_id} processing completed=====")
    print("\n===== Test completed =====")