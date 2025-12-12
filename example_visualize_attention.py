"""
Example: Visualize Attention Weights on Images

This script demonstrates how to visualize attention weights overlaid on actual images.
You can change the token_idx parameter to see different tokens' attention patterns.
"""

from visualize_attention import demo_visualize_token_attention

# Path to your images
image_paths = [
    "examples/llff_flower/images/000.png",
    "examples/llff_flower/images/005.png",
    "examples/llff_flower/images/010.png"
]

# Visualize attention for a specific token
# token_idx: which patch token to visualize (will be auto-adjusted if invalid)
# head_idx: which attention head to visualize (0-15 for VGGT)
# block_idx: which transformer block (0-23 for VGGT)

figures, result = demo_visualize_token_attention(
    image_paths=image_paths,
    block_idx=23,           # Last block (deepest layer)
    token_idx=500,          # Token to visualize (patch index)
    head_idx=0,             # First attention head
    attention_types=['frame', 'global']  # Show both frame and global attention
)

# You can also try different tokens:
# token_idx=100  # Top-left region
# token_idx=500  # Middle region
# token_idx=1000 # Bottom-right region

# Or different heads to see what different heads learn:
# head_idx=0, 1, 2, ..., 15

# Or different blocks to see attention at different depths:
# block_idx=0   # Early layer (more local features)
# block_idx=12  # Middle layer
# block_idx=23  # Deep layer (more semantic features)
