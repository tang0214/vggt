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
from PIL import Image
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


def visualize_attention_on_image(
    attn_weights: torch.Tensor,
    images: torch.Tensor,
    token_idx: int,
    head_idx: int = 0,
    patch_start_idx: int = 5,
    attention_type: str = 'frame',
    layer_name: str = '',
    cmap: str = 'turbo',
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Visualize attention weights overlaid on actual images.

    Args:
        attn_weights: Attention weights tensor
                     - Frame attention: [B*S, num_heads, P+special_tokens, P+special_tokens]
                     - Global attention: [B, num_heads, S*P+special_tokens, S*P+special_tokens]
        images: Original images tensor [B, S, 3, H, W] or [S, 3, H, W]
        token_idx: Which token (patch) to visualize attention from (should be >= patch_start_idx)
        head_idx: Which attention head to visualize
        patch_start_idx: Index where patch tokens start (after camera/register tokens)
        attention_type: 'frame' or 'global'
        layer_name: Name of the layer for the title
        cmap: Colormap to use
        alpha: Transparency of attention overlay
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Ensure images has batch dimension
    if len(images.shape) == 4:
        images = images.unsqueeze(0)  # [B, S, 3, H, W]

    B, S, C, H, W = images.shape

    # Calculate grid dimensions
    patch_size = 14  # VGGT default
    grid_h = H // patch_size
    grid_w = W // patch_size
    num_patches = grid_h * grid_w

    if attention_type == 'frame':
        # Frame attention: [B*S, num_heads, N, N] where N = patches + special tokens
        # Extract attention for the specified token
        attn = attn_weights[0, head_idx, token_idx, :].numpy()  # Use first frame

        # Remove special tokens (only keep patch tokens)
        attn_patches = attn[patch_start_idx:patch_start_idx + num_patches]

        # Re-normalize: original attention was softmax over ALL tokens (including special tokens)
        attn_patches = attn_patches / (attn_patches.sum() + 1e-8)

        # Reshape to spatial grid
        attn_map = attn_patches.reshape(grid_h, grid_w)

        # Visualize on single frame
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Original image
        img_np = images[0, 0].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image (Frame 0)')
        axes[0].axis('off')

        # Attention overlay
        axes[1].imshow(img_np)
        attn_resized = F.interpolate(
            torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        im = axes[1].imshow(attn_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=attn_resized.max())
        axes[1].set_title(f'Frame Attention from Token {token_idx}\n{layer_name}, Head {head_idx}')
        axes[1].axis('off')

        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Attention Weight')

    elif attention_type == 'global':
        # Global attention: [B, num_heads, S*P, S*P] where P = tokens per frame
        # Token layout: [frame0_tokens, frame1_tokens, ..., frameS-1_tokens]
        # Each frame_tokens = [camera, registers, patches]

        attn = attn_weights[0, head_idx, token_idx, :].numpy()

        # Calculate tokens per frame
        total_tokens = attn.shape[0]
        tokens_per_frame = total_tokens // S

        # Extract patch tokens from each frame separately (skip camera & register tokens)
        attn_patches_list = []
        for frame_idx in range(S):
            frame_start = frame_idx * tokens_per_frame
            patches_start = frame_start + patch_start_idx
            patches_end = patches_start + num_patches
            attn_patches_list.append(attn[patches_start:patches_end])

        # Concatenate all patch tokens
        attn_patches = np.concatenate(attn_patches_list)

        # Re-normalize: original attention was softmax over ALL tokens (including special tokens)
        attn_patches = attn_patches / (attn_patches.sum() + 1e-8)

        # Reshape to [S, num_patches]
        attn_per_frame = attn_patches.reshape(S, num_patches)

        # Create subplot for each frame
        fig, axes = plt.subplots(2, S, figsize=(5*S, 10))
        if S == 1:
            axes = axes.reshape(2, 1)

        for frame_idx in range(S):
            # Original image
            img_np = images[0, frame_idx].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[0, frame_idx].imshow(img_np)
            axes[0, frame_idx].set_title(f'Frame {frame_idx}')
            axes[0, frame_idx].axis('off')

            # Attention overlay
            attn_map = attn_per_frame[frame_idx].reshape(grid_h, grid_w)
            axes[1, frame_idx].imshow(img_np)

            attn_resized = F.interpolate(
                torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

            im = axes[1, frame_idx].imshow(attn_resized, cmap=cmap, alpha=alpha,
                                           vmin=0, vmax=attn_per_frame.max())
            axes[1, frame_idx].set_title(f'Attention to Frame {frame_idx}')
            axes[1, frame_idx].axis('off')

        plt.suptitle(f'Global Attention from Token {token_idx}\n{layer_name}, Head {head_idx}',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04, label='Attention Weight')

    plt.tight_layout()
    return fig


def get_attention_values(
    attn_weights: torch.Tensor,
    images: torch.Tensor,
    token_idx: int,
    head_idx: int = 0,
    patch_start_idx: int = 5,
    attention_type: str = 'frame'
) -> Dict:
    """
    Extract attention weight values without visualization.
    Same inputs as visualize_attention_on_image but returns pure numpy arrays.

    Args:
        attn_weights: Attention weights tensor
                     - Frame attention: [B*S, num_heads, P+special_tokens, P+special_tokens]
                     - Global attention: [B, num_heads, S*P+special_tokens, S*P+special_tokens]
        images: Original images tensor [B, S, 3, H, W] or [S, 3, H, W]
        token_idx: Which token (patch) to visualize attention from (should be >= patch_start_idx)
        head_idx: Which attention head to extract
        patch_start_idx: Index where patch tokens start (after camera/register tokens)
        attention_type: 'frame' or 'global'

    Returns:
        Dictionary containing:
            - 'attention_maps': numpy array of attention maps
                - Frame: [grid_h, grid_w] - single frame attention map
                - Global: [S, grid_h, grid_w] - attention map for each frame
            - 'attention_maps_resized': numpy array resized to image resolution
                - Frame: [H, W]
                - Global: [S, H, W]
            - 'images': numpy array of images [S, H, W, 3] in range [0, 1]
            - 'metadata': dict with grid_h, grid_w, H, W, S, token_idx, head_idx
    """
    # Ensure images has batch dimension
    if len(images.shape) == 4:
        images = images.unsqueeze(0)  # [B, S, 3, H, W]

    B, S, C, H, W = images.shape

    # Calculate grid dimensions
    patch_size = 14  # VGGT default
    grid_h = H // patch_size
    grid_w = W // patch_size
    num_patches = grid_h * grid_w

    # Convert images to numpy [S, H, W, 3]
    images_np = images[0].permute(0, 2, 3, 1).cpu().numpy()
    images_np = np.clip(images_np, 0, 1)

    result = {
        'images': images_np,
        'metadata': {
            'grid_h': grid_h,
            'grid_w': grid_w,
            'H': H,
            'W': W,
            'S': S,
            'token_idx': token_idx,
            'head_idx': head_idx,
            'attention_type': attention_type,
            'patch_start_idx': patch_start_idx
        }
    }

    if attention_type == 'frame':
        # Frame attention: [B*S, num_heads, N, N] where N = patches + special tokens
        # Extract attention for the specified token
        attn = attn_weights[0, head_idx, token_idx, :].numpy()  # Use first frame

        # Remove special tokens (only keep patch tokens)
        attn_patches = attn[patch_start_idx:patch_start_idx + num_patches]

        # Re-normalize: original attention was softmax over ALL tokens (including special tokens)
        attn_patches = attn_patches / (attn_patches.sum() + 1e-8)

        # Reshape to spatial grid [grid_h, grid_w]
        attn_map = attn_patches.reshape(grid_h, grid_w)

        # Resize to image resolution [H, W]
        attn_resized = F.interpolate(
            torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        result['attention_maps'] = attn_map
        result['attention_maps_resized'] = attn_resized

    elif attention_type == 'global':
        # Global attention: [B, num_heads, S*P, S*P] where P = tokens per frame
        # Token layout: [frame0_tokens, frame1_tokens, ..., frameS-1_tokens]
        # Each frame_tokens = [camera, registers, patches]

        attn = attn_weights[0, head_idx, token_idx, :].numpy()

        # Calculate tokens per frame
        total_tokens = attn.shape[0]
        tokens_per_frame = total_tokens // S

        # Extract patch tokens from each frame separately (skip camera & register tokens)
        attn_patches_list = []
        for frame_idx in range(S):
            frame_start = frame_idx * tokens_per_frame
            patches_start = frame_start + patch_start_idx
            patches_end = patches_start + num_patches
            attn_patches_list.append(attn[patches_start:patches_end])

        # Concatenate all patch tokens
        attn_patches = np.concatenate(attn_patches_list)

        # Re-normalize: original attention was softmax over ALL tokens (including special tokens)
        attn_patches = attn_patches / (attn_patches.sum() + 1e-8)

        # Reshape to [S, num_patches]
        attn_per_frame = attn_patches.reshape(S, num_patches)

        # Reshape each frame to spatial grid [S, grid_h, grid_w]
        attn_maps = np.array([attn_per_frame[i].reshape(grid_h, grid_w) for i in range(S)])

        # Resize to image resolution [S, H, W]
        attn_resized_list = []
        for frame_idx in range(S):
            attn_resized = F.interpolate(
                torch.from_numpy(attn_maps[frame_idx]).unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            attn_resized_list.append(attn_resized)

        attn_resized = np.array(attn_resized_list)  # [S, H, W]

        result['attention_maps'] = attn_maps  # [S, grid_h, grid_w]
        result['attention_maps_resized'] = attn_resized  # [S, H, W]

    return result


def get_all_frames_attention(
    attn_weights: torch.Tensor,
    images: torch.Tensor,
    token_idx: int,
    head_idx: int = 0,
    patch_start_idx: int = 5,
    frame_indices: Optional[List[int]] = None
) -> Dict:
    """
    Extract frame attention maps for multiple frames.

    Args:
        attn_weights: Frame attention weights [B*S, num_heads, N, N]
        images: Images tensor [B, S, 3, H, W] or [S, 3, H, W]
        token_idx: Token index to extract attention from
        head_idx: Attention head index
        patch_start_idx: Index where patch tokens start
        frame_indices: List of frame indices to extract, None means all frames

    Returns:
        Dictionary containing:
            - 'attention_maps': numpy array [num_frames, grid_h, grid_w]
            - 'attention_maps_resized': numpy array [num_frames, H, W]
            - 'images': numpy array [num_frames, H, W, 3]
            - 'frame_indices': list of extracted frame indices
            - 'metadata': dict with grid_h, grid_w, H, W, S, token_idx, head_idx
    """
    # Ensure images has batch dimension
    if len(images.shape) == 4:
        images = images.unsqueeze(0)

    B, S, C, H, W = images.shape

    # If frame_indices not specified, use all frames
    if frame_indices is None:
        frame_indices = list(range(S))

    # Calculate grid dimensions
    patch_size = 14
    grid_h = H // patch_size
    grid_w = W // patch_size
    num_patches = grid_h * grid_w

    # Convert images to numpy
    images_np = images[0].permute(0, 2, 3, 1).cpu().numpy()
    images_np = np.clip(images_np, 0, 1)

    results = {
        'attention_maps': [],
        'attention_maps_resized': [],
        'images': images_np,
        'frame_indices': frame_indices,
        'metadata': {
            'grid_h': grid_h,
            'grid_w': grid_w,
            'H': H,
            'W': W,
            'S': S,
            'token_idx': token_idx,
            'head_idx': head_idx,
            'patch_start_idx': patch_start_idx,
            'attention_type': 'frame'
        }
    }

    # Extract attention for each frame
    for frame_idx in frame_indices:
        # Extract attention for this frame
        attn = attn_weights[frame_idx, head_idx, token_idx, :].numpy()

        # Remove special tokens
        attn_patches = attn[patch_start_idx:patch_start_idx + num_patches]

        # Re-normalize: original attention was softmax over ALL tokens (including special tokens)
        attn_patches = attn_patches / (attn_patches.sum() + 1e-8)

        # Reshape to spatial grid
        attn_map = attn_patches.reshape(grid_h, grid_w)

        # Resize to image resolution
        attn_resized = F.interpolate(
            torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        results['attention_maps'].append(attn_map)
        results['attention_maps_resized'].append(attn_resized)

    # Convert to numpy arrays
    results['attention_maps'] = np.array(results['attention_maps'])
    results['attention_maps_resized'] = np.array(results['attention_maps_resized'])

    return results


def visualize_frame_attention(
    attn_weights: torch.Tensor,
    images: torch.Tensor,
    token_idx: int,
    head_idx: int = 0,
    patch_start_idx: int = 5,
    frame_idx: int = 0,
    layer_name: str = '',
    cmap: str = 'turbo',
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Visualize frame attention for a specific frame.

    Args:
        attn_weights: Frame attention weights [B*S, num_heads, N, N]
        images: Images tensor [B, S, 3, H, W] or [S, 3, H, W]
        token_idx: Token index to visualize
        head_idx: Attention head index
        patch_start_idx: Index where patch tokens start
        frame_idx: Which frame to visualize (0 to S-1)
        layer_name: Name of the layer for title
        cmap: Colormap to use
        alpha: Transparency of attention overlay
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if len(images.shape) == 4:
        images = images.unsqueeze(0)

    B, S, C, H, W = images.shape

    # Validate frame_idx
    if frame_idx >= S or frame_idx < 0:
        raise ValueError(f"frame_idx {frame_idx} is out of range [0, {S-1}]")

    # Calculate grid dimensions
    patch_size = 14
    grid_h = H // patch_size
    grid_w = W // patch_size
    num_patches = grid_h * grid_w

    # Extract attention for specified frame
    attn = attn_weights[frame_idx, head_idx, token_idx, :].numpy()
    attn_patches = attn[patch_start_idx:patch_start_idx + num_patches]

    # Re-normalize: original attention was softmax over ALL tokens (including special tokens)
    attn_patches = attn_patches / (attn_patches.sum() + 1e-8)

    attn_map = attn_patches.reshape(grid_h, grid_w)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original image
    img_np = images[0, frame_idx].permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original Image (Frame {frame_idx})')
    axes[0].axis('off')

    # Attention overlay
    axes[1].imshow(img_np)
    attn_resized = F.interpolate(
        torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    im = axes[1].imshow(attn_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=attn_resized.max())
    axes[1].set_title(f'Frame Attention from Token {token_idx}\n{layer_name}, Head {head_idx}, Frame {frame_idx}')
    axes[1].axis('off')

    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Attention Weight')
    plt.tight_layout()

    return fig


def visualize_all_frames_attention(
    attn_weights: torch.Tensor,
    images: torch.Tensor,
    token_idx: int,
    head_idx: int = 0,
    patch_start_idx: int = 5,
    frame_indices: Optional[List[int]] = None,
    layer_name: str = '',
    cmap: str = 'turbo',
    alpha: float = 0.6,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Visualize frame attention for all specified frames in a single figure.

    Args:
        attn_weights: Frame attention weights [B*S, num_heads, N, N]
        images: Images tensor [B, S, 3, H, W] or [S, 3, H, W]
        token_idx: Token index to visualize
        head_idx: Attention head index
        patch_start_idx: Index where patch tokens start
        frame_indices: List of frame indices to visualize, None means all frames
        layer_name: Name of the layer for title
        cmap: Colormap to use
        alpha: Transparency of attention overlay
        figsize: Figure size (auto-calculated if None)

    Returns:
        matplotlib Figure object
    """
    # Get all frames attention data
    all_frames_data = get_all_frames_attention(
        attn_weights=attn_weights,
        images=images,
        token_idx=token_idx,
        head_idx=head_idx,
        patch_start_idx=patch_start_idx,
        frame_indices=frame_indices
    )

    frame_indices = all_frames_data['frame_indices']
    num_frames = len(frame_indices)

    # Auto-calculate figure size if not specified
    if figsize is None:
        figsize = (5 * num_frames, 10)

    # Create visualization
    fig, axes = plt.subplots(2, num_frames, figsize=figsize)
    if num_frames == 1:
        axes = axes.reshape(2, 1)

    for i, frame_idx in enumerate(frame_indices):
        # Original image
        img_np = all_frames_data['images'][frame_idx]
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'Frame {frame_idx}')
        axes[0, i].axis('off')

        # Attention overlay
        axes[1, i].imshow(img_np)
        attn_resized = all_frames_data['attention_maps_resized'][i]
        im = axes[1, i].imshow(
            attn_resized,
            cmap=cmap,
            alpha=alpha,
            vmin=0,
            vmax=all_frames_data['attention_maps_resized'].max()
        )
        axes[1, i].set_title(f'Frame Attention (Frame {frame_idx})')
        axes[1, i].axis('off')

    plt.suptitle(
        f'Frame Attention from Token {token_idx}\n{layer_name}, Head {head_idx}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    plt.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04, label='Attention Weight')
    plt.tight_layout()

    return fig


def extract_both_attentions(
    model: VGGT,
    images: torch.Tensor,
    block_idx: int = 23,
    dtype=torch.float16
) -> Dict:
    """
    Extract both frame and global attention for a specific block.

    Args:
        model: VGGT model
        images: Input images [B, S, 3, H, W]
        block_idx: Which block to extract from
        dtype: Data type for inference

    Returns:
        Dictionary containing both attention maps and metadata
    """
    extractor = AttentionExtractor(model)

    # Register hooks for both frame and global attention
    extractor.register_hooks(
        block_indices=[block_idx],
        attention_types=['frame', 'global']
    )

    # Forward pass
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            _ = model(images)

    # Get attention maps
    attention_maps = extractor.get_attention_maps()

    # Clean up
    extractor.clear_hooks()

    # Extract data
    frame_layer = f"frame_block_{block_idx}"
    global_layer = f"global_block_{block_idx}"

    result = {
        'frame_attention': attention_maps.get(frame_layer, None),
        'global_attention': attention_maps.get(global_layer, None),
        'block_idx': block_idx,
        'patch_start_idx': model.aggregator.patch_start_idx
    }

    return result


def demo_visualize_token_attention(
    image_paths: List[str],
    block_idx: int = 23,
    token_idx: int = 500,
    head_idx: int = 0,
    attention_types: List[str] = ['frame', 'global']
):
    """
    Demo: Visualize a specific token's attention on actual images.
    Shows both frame and global attention.

    Args:
        image_paths: List of image file paths
        block_idx: Which block to visualize (0-23)
        token_idx: Which token (patch) to visualize attention from
        head_idx: Which attention head to visualize
        attention_types: Which attention types to show
    """
    print("=" * 80)
    print("VGGT Token Attention Visualization on Images")
    print("=" * 80)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Device: {device}")
    print(f"Loading model...")

    # Load model
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    print(f"Loading {len(image_paths)} images...")
    images = load_and_preprocess_images(image_paths).to(device)

    # Extract both attentions
    print(f"Extracting attention from block {block_idx}...")
    result = extract_both_attentions(model, images, block_idx=block_idx, dtype=dtype)

    patch_start_idx = result['patch_start_idx']

    # Calculate valid token range
    if len(images.shape) == 4:
        images_vis = images.unsqueeze(0)
    else:
        images_vis = images

    S = images_vis.shape[1]  # Number of frames
    H, W = images_vis.shape[3], images_vis.shape[4]
    grid_h, grid_w = H // 14, W // 14
    num_patches = grid_h * grid_w

    print(f"\nImage info:")
    print(f"  Number of frames: {S}")
    print(f"  Image size: {H}x{W}")
    print(f"  Grid size: {grid_h}x{grid_w}")
    print(f"  Patches per frame: {num_patches}")
    print(f"  Special tokens before patches: {patch_start_idx}")
    print(f"  Valid token range for visualization: {patch_start_idx} to {patch_start_idx + num_patches - 1}")

    # Adjust token_idx if needed
    if token_idx < patch_start_idx:
        print(f"\nWarning: token_idx {token_idx} is a special token (< {patch_start_idx})")
        token_idx = patch_start_idx
        print(f"Adjusted to first patch token: {token_idx}")
    elif token_idx >= patch_start_idx + num_patches:
        print(f"\nWarning: token_idx {token_idx} is out of range")
        token_idx = patch_start_idx + num_patches // 2
        print(f"Adjusted to middle patch token: {token_idx}")

    figures = []

    # Visualize frame attention
    if 'frame' in attention_types and result['frame_attention'] is not None:
        print(f"\nVisualizing frame attention...")
        frame_attn = result['frame_attention']
        print(f"  Frame attention shape: {frame_attn['attn_weights'].shape}")

        fig = visualize_attention_on_image(
            attn_weights=frame_attn['attn_weights'],
            images=images_vis,
            token_idx=token_idx,
            head_idx=head_idx,
            patch_start_idx=patch_start_idx,
            attention_type='frame',
            layer_name=f"Frame Block {block_idx}"
        )
        figures.append(('frame', fig))

    # Visualize global attention
    if 'global' in attention_types and result['global_attention'] is not None:
        print(f"\nVisualizing global attention...")
        global_attn = result['global_attention']
        print(f"  Global attention shape: {global_attn['attn_weights'].shape}")

        fig = visualize_attention_on_image(
            attn_weights=global_attn['attn_weights'],
            images=images_vis,
            token_idx=token_idx,
            head_idx=head_idx,
            patch_start_idx=patch_start_idx,
            attention_type='global',
            layer_name=f"Global Block {block_idx}"
        )
        figures.append(('global', fig))

    print(f"\nShowing {len(figures)} visualizations...")
    plt.show()

    return figures, result


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

    print("=" * 80)
    print("NEW: Visualize Token Attention on Images")
    print("=" * 80)
    print("This shows attention overlaid on actual images for both frame and global attention\n")

    # NEW: Visualize attention on actual images
    figures, result = demo_visualize_token_attention(
        image_paths,
        block_idx=23,
        token_idx=500,  # Will be auto-adjusted if invalid
        head_idx=0,
        attention_types=['frame', 'global']  # Show both
    )

    print("\n" + "=" * 80)
    print("OLD Mode 1: Basic Visualization (token-to-token heatmap)")
    print("=" * 80)
    demo_single_layer_attention(
        image_paths,
        block_idx=23,
        attention_type='global',
        token_idx=500,
        head_idx=0,
        mode='visualize'
    )

    print("\n" + "=" * 80)
    print("OLD Mode 2: Data extraction")
    print("=" * 80)
    data = demo_single_layer_attention(
        image_paths,
        block_idx=23,
        attention_type='global',
        token_idx=500,
        head_idx=0,
        mode='data'
    )
    print(f"Returned data keys: {data.keys()}")
