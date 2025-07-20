import os
import json
from typing import List, Optional, Tuple, Dict

# Set environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"  # Prefer to use xformers' memory_efficient_attention_forward function

# Import necessary libraries
import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Initialize the large model
# llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.5, enforce_eager=True, max_model_len=10000)
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
    """Load user historical purchase data and candidates"""
    user_dir = os.path.join(os.path.dirname(__file__),"../../", "dataset", user_id)

    # Read historical purchase records
    history_path = os.path.join(user_dir, "history.json")
    with open(history_path, 'r') as f:
        history_data = json.load(f)

    # Read candidates
    candidate_path = os.path.join(user_dir, "candidate.json")
    with open(candidate_path, 'r') as f:
        candidate_data = json.load(f)

    return history_data, candidate_data

def visualize_last_token_attention_gradient(
    attn_weights: torch.Tensor,
    tokens: List[str],
    output_path: str = "last_token_attention_gradient.png",
    head_idx: Optional[int] = None,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
) -> Dict[str, str]:
    """Visualize the attention score density distribution of the last token to verify attention sparsity

    Args:
        attn_weights: Attention weight tensor, shape [num_heads, seq_len, seq_len]
        tokens: Token list of input sequence
        output_path: Output image path
        head_idx: Attention head index to visualize, None means use first head (head 0)
        plot_title: Optional plot title, if None use default title
        figsize: Plot size
        dpi: Plot resolution

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
    else:
        # Use first head as default
        attn = attn_weights[0].to(torch.float32).numpy()

    seq_len = attn.shape[0]

    # Ensure tokens length matches attention matrix
    if len(tokens) != seq_len:
        print(f"Warning: tokens length ({len(tokens)}) does not match attention matrix sequence length ({seq_len})")
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        else:
            tokens = tokens + [""] * (seq_len - len(tokens))

    # Extract attention scores from the last token to all previous tokens
    # attn[-1, :] gives us the attention scores of the last token (query) to all tokens (keys)
    last_token_attention = attn[-1, :]  # Shape: [seq_len]

    # Create the visualization
    plt.figure(figsize=figsize)

    # Sort attention scores in descending order to show distribution
    sorted_attention = np.sort(last_token_attention)[::-1]  # Sort in descending order

    # Create token proportion array (0 to 1)
    token_proportions = np.linspace(0, 1, len(sorted_attention))

    # Create density gradient curve
    plt.plot(token_proportions, sorted_attention, 'b-', linewidth=2, alpha=0.8)
    plt.fill_between(token_proportions, sorted_attention, alpha=0.3, color='skyblue')

    # Set labels and title
    plt.xlabel('Token Proportion', fontsize=12)
    plt.ylabel('Attention Score', fontsize=12)

    # Extract head number for title
    if head_idx is not None:
        title = f'Attention Score Distribution (head {head_idx})'
    else:
        title = 'Attention Score Distribution (head 0)'

    plt.title(plot_title if plot_title else title, fontsize=14)
    plt.grid(True, alpha=0.3)

    # Set x-axis to 0-1 range
    plt.xlim(0, 1)
    plt.ylim(0, max(sorted_attention) * 1.05)

    # Save image
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    output_paths = {'gradient': output_path}
    print(f"Last token attention density distribution visualization saved to {output_path}")
    print(f"Attention sparsity: Top 10% tokens have attention >= {sorted_attention[int(0.1 * len(sorted_attention))]:.4f}")
    print(f"Attention sparsity: Top 20% tokens have attention >= {sorted_attention[int(0.2 * len(sorted_attention))]:.4f}")

    return output_paths

def generate_recommendation_with_last_token_attention(user_id):
    """
    Generate recommendations for specified user and visualize last token attention weights

    Args:
        user_id: User ID

    Returns:
        Generated output result
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
{{"rank": "[] > [] .. > []"}} Do not say any word except this JSON format."""

    # Create an instance of the simplified PromptFieldTracker
    tracker = PromptFieldTracker(tokenizer)

    print(f"Number of loaded history records: {len(history_data)}")
    print(f"Number of loaded candidate items: {len(candidate_data)}")

    # Get complete prompt token IDs
    input_ids_from_tracker, _ = tracker.track_positions(prefix_prompt, all_items, query_prompt)

    # Decode token IDs to complete prompt string
    input_prompt = tokenizer.decode(input_ids_from_tracker)

    # Get cache_metadata reference
    cache_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_metadata

    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1)

    # Set cache metadata
    cache_metadata["check"] = False
    cache_metadata['collect'] = False

    # Enable attention weights return - this is the key step to get real attention weights
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.return_attn_weights = True

    # Generate output
    output = llm.generate([input_prompt], sampling_params)

    # Get attention weights
    attn_weights = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.first_layer_attn_weights

    # Add debug information
    print(f"Input prompt length: {len(input_prompt)}")
    print(f"Input IDs length: {len(input_ids_from_tracker)}")

    # If unable to get attention weights, try to get from other positions of the model
    if attn_weights is None:
        print("Attempt to obtain attention weights from other positions of the model...")
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn'):
                if hasattr(layer.self_attn.attn, 'last_attn_weights'):
                    attn_weights = layer.self_attn.attn.last_attn_weights
                    print(f"Obtained attention weights from the {i}th layer")
                    break

    # Add more debug information
    if attn_weights is not None:
        print(f"Raw attention weights shape: {attn_weights.shape}")
        print(f"Raw attention weights device: {attn_weights.device}")
        print(f"Raw attention weights dtype: {attn_weights.dtype}")
    else:
        print("Warning: No attention weights obtained from any layer")

    # If successfully obtained attention weights, perform visualization
    if attn_weights is not None:
        print("Successfully obtained attention weights, generating last token attention visualization...")
        # Get input token list
        tokens = tokenizer.convert_ids_to_tokens(input_ids_from_tracker)
        # Convert attention weights to float32 type to avoid BFloat16 incompatibility issues
        attn_weights = attn_weights.to(torch.float32)

        print(f"Attention weight shape: {attn_weights.shape}")

        # Ensure attention weights have correct shape [num_heads, seq_len, seq_len]
        if len(attn_weights.shape) == 3:
            _, seq_len_q, _ = attn_weights.shape

            # Check if sequence length is valid
            if seq_len_q == 0:
                print("Error: Attention weights have zero sequence length, cannot generate visualization")
                return output

            # Check if token list length matches attention weights sequence length
            if len(tokens) != seq_len_q:
                # If not matching, may need to truncate or pad token list
                if len(tokens) > seq_len_q:
                    tokens = tokens[:seq_len_q]
                    print(f"The tokens list has been truncated to {len(tokens)} tokens.")
                else:
                    # If token list is too short, pad with empty strings
                    tokens = tokens + [""] * (seq_len_q - len(tokens))
                    print(f"The tokens list has been filled to {len(tokens)} tokens.")

            # Get sequence length
            seq_len = attn_weights.shape[1]
            print(f"Sequence Length: {seq_len}")

            # Check if sequence length is sufficient for visualization
            if seq_len <= 0:
                print("Error: Invalid sequence length for visualization")
                return output

            # Use real attention weight data for visualization
            # Visualize multiple attention heads
            for head_idx in [0, 5, 15]:  # Visualize heads 0, 5, and 15
                if head_idx < attn_weights.shape[0]:  # Ensure head index is valid
                    _ = visualize_last_token_attention_gradient(
                        attn_weights=attn_weights,
                        tokens=tokens,
                        output_path=f"last_token_attention_{user_id}_head_{head_idx}.png",
                        head_idx=head_idx
                    )
                    print(f"Generated visualization for head {head_idx}")
                else:
                    print(f"Head {head_idx} does not exist (total heads: {attn_weights.shape[0]})")
        else:
            print(f"Error: The attention weight shape is incorrect. {attn_weights.shape}, Should be [num_heads, seq_len, seq_len]")
    else:
        print("Error: Unable to obtain attention weights, unable to generate visualization")

    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")

    return output

if __name__ == "__main__":
    # Default user ID
    user_id = "user_A10FW892S59ABJ"

    print(f"\n===== Generate recommendations and visualize last token attention for user {user_id} =====")

    try:
        # Generate recommendations and get last token attention weight visualization
        output = generate_recommendation_with_last_token_attention(user_id)
        print(f"Recommendation generation completed")

        # Get attention weights for additional analysis if needed
        attn_weights = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.first_layer_attn_weights

        # If successfully obtained attention weights, perform additional analysis
        if attn_weights is not None:
            print(f"Final attention weights shape: {attn_weights.shape}")
            print(f"Number of attention heads: {attn_weights.shape[0]}")
            print(f"Sequence length: {attn_weights.shape[1]}")

            # Analyze last token attention statistics across all heads
            # Convert to float32 first to avoid BFloat16 conversion issues
            last_token_attn_all_heads = attn_weights[:, -1, :].to(torch.float32).cpu().numpy()  # Shape: [num_heads, seq_len]

            print(f"\nLast token attention statistics across all heads:")
            print(f"Max attention score: {last_token_attn_all_heads.max():.4f}")
            print(f"Min attention score: {last_token_attn_all_heads.min():.4f}")
            print(f"Mean attention score: {last_token_attn_all_heads.mean():.4f}")
            print(f"Std attention score: {last_token_attn_all_heads.std():.4f}")

            # Find positions with highest attention scores
            max_positions = []
            for head_idx in range(min(3, attn_weights.shape[0])):  # Check first 3 heads
                head_attention = last_token_attn_all_heads[head_idx]
                max_pos = np.argmax(head_attention)
                max_score = head_attention[max_pos]
                max_positions.append((head_idx, max_pos, max_score))
                print(f"Head {head_idx}: Max attention at position {max_pos} with score {max_score:.4f}")

    except Exception as e:
        print(f"An error occurred while generating recommendations for user {user_id}: {e}")
        import traceback
        traceback.print_exc()

    print(f"===== User {user_id} processing completed =====")
    print("\n===== Test completed =====")
