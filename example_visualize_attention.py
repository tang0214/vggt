"""
Example: Visualize Attention Weights on Images

This script demonstrates how to visualize attention weights overlaid on actual images.
You can change the token_idx parameter to see different tokens' attention patterns.

Also includes an example of extracting pure attention values for custom processing.
"""

from visualize_attention import demo_visualize_token_attention, get_attention_values, extract_both_attentions
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import torch
import numpy as np

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


# ============================================================
# Example: Extract pure attention values (no visualization)
# ============================================================
print("\n" + "="*60)
print("Example: Extracting Pure Attention Values")
print("="*60)

# Load model and images
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

images = load_and_preprocess_images(image_paths).to(device)

# Extract both frame and global attention
result = extract_both_attentions(model, images, block_idx=23, dtype=dtype)

# Get pure attention values (numpy arrays)
frame_values = get_attention_values(
    attn_weights=result['frame_attention']['attn_weights'],
    images=images,
    token_idx=500,
    head_idx=0,
    patch_start_idx=result['patch_start_idx'],
    attention_type='frame'
)

global_values = get_attention_values(
    attn_weights=result['global_attention']['attn_weights'],
    images=images,
    token_idx=500,
    head_idx=0,
    patch_start_idx=result['patch_start_idx'],
    attention_type='global'
)

# Print information about extracted values
print("\n--- Frame Attention ---")
print(f"Attention map shape (patch grid): {frame_values['attention_maps'].shape}")
print(f"Attention map shape (resized): {frame_values['attention_maps_resized'].shape}")
print(f"Attention range: [{frame_values['attention_maps_resized'].min():.4f}, {frame_values['attention_maps_resized'].max():.4f}]")

print("\n--- Global Attention ---")
print(f"Attention map shape (patch grid): {global_values['attention_maps'].shape}")
print(f"Attention map shape (resized): {global_values['attention_maps_resized'].shape}")
for i in range(len(global_values['attention_maps'])):
    attn_frame = global_values['attention_maps_resized'][i]
    print(f"  Frame {i} attention range: [{attn_frame.min():.4f}, {attn_frame.max():.4f}]")

# Save to .npy files
np.save('frame_attention.npy', frame_values['attention_maps_resized'])
np.save('global_attention.npy', global_values['attention_maps_resized'])
print("\n✓ Saved attention maps to frame_attention.npy and global_attention.npy")

# You can now do custom processing, analysis, or visualization with these values
print("\nYou can now use these numpy arrays for:")
print("  - Custom visualization")
print("  - Statistical analysis")
print("  - Save for later use")
print("  - Compare attention across different models/layers")
print("  - And more!")
