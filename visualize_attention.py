"""
VGGT Attention Map Visualization Tool

This script extracts and visualizes attention maps from VGGT model layers.
Supports two modes:
1. Visualization mode: Generate heatmaps like in the paper
2. Data mode: Return raw attention weight values
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


class AttentionExtractor:
    """Extract attention maps from VGGT model layers."""

    def __init__(self, model: VGGT):
        self.model = model
        self.attention_maps = {}
        self.hooks = []

    def _create_attention_hook(self, layer_name: str, layer_type: str):
        """Create a hook to capture attention weights from a layer.

        Args:
            layer_name: Name identifier for the layer (e.g., "global_block_0")
            layer_type: Type of attention block ("global" or "frame")
        """
        def hook_fn(module, input, output):
            # We need to manually compute attention to get the weights
            # Since the model uses fused_attn by default
            x = input[0]  # Input tensor
            pos = input[1] if len(input) > 1 else None

            # Get the attention module
            attn = module.attn
            B, N, C = x.shape

            # Compute QKV
            x_norm = module.norm1(x)
            qkv = attn.qkv(x_norm).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn.q_norm(q), attn.k_norm(k)

            # Apply RoPE if needed
            if attn.rope is not None and pos is not None:
                q = attn.rope(q, pos)
                k = attn.rope(k, pos)

            # Compute attention weights manually
            q = q * attn.scale
            attn_weights = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
            attn_weights = attn_weights.softmax(dim=-1)

            # Store attention weights
            self.attention_maps[layer_name] = {
                'attn_weights': attn_weights.detach().cpu(),  # [B, num_heads, N, N]
                'layer_type': layer_type,
                'num_heads': attn.num_heads,
                'num_tokens': N
            }

        return hook_fn

    def register_hooks(self, block_indices: Optional[List[int]] = None,
                      attention_types: List[str] = ['global', 'frame']):
        """Register hooks to specified blocks.

        Args:
            block_indices: List of block indices to hook. If None, hook all blocks.
            attention_types: Types of attention to capture ('global', 'frame', or both)
        """
        # Clear existing hooks
        self.clear_hooks()
        self.attention_maps = {}

        aggregator = self.model.aggregator

        if block_indices is None:
            block_indices = list(range(len(aggregator.global_blocks)))

        for idx in block_indices:
            if 'global' in attention_types and idx < len(aggregator.global_blocks):
                hook = aggregator.global_blocks[idx].register_forward_hook(
                    self._create_attention_hook(f"global_block_{idx}", "global")
                )
                self.hooks.append(hook)

            if 'frame' in attention_types and idx < len(aggregator.frame_blocks):
                hook = aggregator.frame_blocks[idx].register_forward_hook(
                    self._create_attention_hook(f"frame_block_{idx}", "frame")
                )
                self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_attention_maps(self) -> Dict:
        """Return the collected attention maps."""
        return self.attention_maps


def visualize_attention_map(
    attn_weights: torch.Tensor,
    token_idx: int = 0,
    head_idx: int = 0,
    frame_idx: int = 0,
    title: str = "Attention Map",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'turbo'
) -> plt.Figure:
    """Visualize a single attention map.

    Args:
        attn_weights: Attention weights tensor [B, num_heads, N, N]
        token_idx: Which token's attention to visualize (query token)
        head_idx: Which attention head to visualize
        frame_idx: Which frame/batch to visualize
        title: Plot title
        figsize: Figure size
        cmap: Colormap to use

    Returns:
        matplotlib Figure object
    """
    # Extract attention for specific token and head
    # Shape: [N] - attention weights from token_idx to all other tokens
    attn = attn_weights[frame_idx, head_idx, token_idx, :].numpy()

    num_tokens = len(attn)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Reshape attention to 2D for better visualization
    # Try to make it roughly square
    grid_size = int(np.ceil(np.sqrt(num_tokens)))  # Use ceil to ensure grid_size^2 >= num_tokens

    # Pad to square if needed
    pad_size = grid_size * grid_size
    attn_padded = np.zeros(pad_size)
    attn_padded[:num_tokens] = attn
    attn_2d = attn_padded.reshape(grid_size, grid_size)

    # Plot
    im = ax.imshow(attn_2d, cmap=cmap, aspect='auto')
    ax.set_title(f"{title}\nToken {token_idx}, Head {head_idx}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Key Token Position")
    ax.set_ylabel("Key Token Position")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)

    plt.tight_layout()
    return fig


def demo_single_layer_attention(
    image_paths: List[str],
    block_idx: int = 23,
    attention_type: str = 'global',
    token_idx: int = 500,
    head_idx: int = 0,
    mode: str = 'visualize'
):
    """Demo: Extract and visualize attention from a single layer.

    Args:
        image_paths: List of image file paths
        block_idx: Which block to visualize (0-23)
        attention_type: 'global' or 'frame'
        token_idx: Which token to visualize
        head_idx: Which attention head to visualize
        mode: 'visualize' to show plot, 'data' to return raw values

    Returns:
        If mode='visualize': shows plot
        If mode='data': returns attention weights dict
    """
    print("=" * 60)
    print("VGGT Attention Map Extraction Demo")
    print("=" * 60)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Device: {device}")
    print(f"Loading model...")

    # Load model
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    print(f"Loading images: {len(image_paths)} images")
    images = load_and_preprocess_images(image_paths).to(device)

    # Create attention extractor
    extractor = AttentionExtractor(model)

    # Register hook for specific block
    print(f"Registering hook for {attention_type}_block_{block_idx}")
    extractor.register_hooks(
        block_indices=[block_idx],
        attention_types=[attention_type]
    )

    # Forward pass to collect attention
    print("Running forward pass...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            _ = model(images)

    # Get attention maps
    attention_maps = extractor.get_attention_maps()
    layer_name = f"{attention_type}_block_{block_idx}"

    if layer_name not in attention_maps:
        print(f"Error: Could not extract attention from {layer_name}")
        return None

    attn_data = attention_maps[layer_name]
    attn_weights = attn_data['attn_weights']

    print(f"\nExtracted attention map:")
    print(f"  Shape: {attn_weights.shape}")
    print(f"  [Batch, Num_Heads, Num_Tokens, Num_Tokens]")
    print(f"  Batch size: {attn_weights.shape[0]}")
    print(f"  Number of heads: {attn_data['num_heads']}")
    print(f"  Number of tokens: {attn_data['num_tokens']}")

    # Clean up hooks
    extractor.clear_hooks()

    if mode == 'visualize':
        # Validate and adjust token_idx if necessary
        num_tokens = attn_data['num_tokens']
        if token_idx >= num_tokens:
            print(f"\nWarning: token_idx {token_idx} >= num_tokens {num_tokens}")
            token_idx = num_tokens - 1
            print(f"Adjusted to token_idx = {token_idx}")

        print(f"\nVisualizing attention for token {token_idx}, head {head_idx}...")
        fig = visualize_attention_map(
            attn_weights,
            token_idx=token_idx,
            head_idx=head_idx,
            frame_idx=0,
            title=f"{attention_type.capitalize()} Block {block_idx}"
        )
        plt.show()
        return fig

    elif mode == 'data':
        # Validate token_idx
        num_tokens = attn_data['num_tokens']
        if token_idx >= num_tokens:
            print(f"\nWarning: token_idx {token_idx} >= num_tokens {num_tokens}")
            print(f"Returning data anyway - please use a valid token_idx for indexing")

        print(f"\nReturning raw attention data...")
        return {
            'attention_weights': attn_weights,
            'layer_name': layer_name,
            'layer_type': attention_type,
            'num_heads': attn_data['num_heads'],
            'num_tokens': attn_data['num_tokens']
        }


if __name__ == "__main__":
    # Example usage
    image_paths = [
        "examples/llff_flower/images/000.png",
        "examples/llff_flower/images/005.png",
        "examples/llff_flower/images/010.png"
    ]

    print("Mode 1: Visualization")
    demo_single_layer_attention(
        image_paths,
        block_idx=23,
        attention_type='global',
        token_idx=500,
        head_idx=0,
        mode='visualize'
    )

    print("\n" + "=" * 60)
    print("Mode 2: Data extraction")
    data = demo_single_layer_attention(
        image_paths,
        block_idx=23,
        attention_type='global',
        token_idx=500,
        head_idx=0,
        mode='data'
    )
    print(f"Returned data keys: {data.keys()}")
