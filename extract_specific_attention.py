"""
提取特定 frame、block、token 的 attention values 和 visualization
"""
from visualize_attention import (
    extract_both_attentions,
    visualize_attention_on_image,
    get_attention_values
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import torch
import numpy as np

# ============ 參數設定 ============
image_paths = [
    "examples/kitchen/images/00.png",
    "examples/kitchen/images/01.png",
    "examples/kitchen/images/02.png"
]

# 你的需求
target_frame_number = 2      # 第幾個frame (1-based，使用者友好)
target_block_number = 20     # 第幾個block (1-based)
target_token_position = 820  # 在該frame中的第幾個token (0-based)

# 轉換為 0-based index
frame_idx = target_frame_number - 1  # 第2個frame -> index 1
block_idx = target_block_number - 1  # 第20個block -> index 19
head_idx = 0  # 使用第0個attention head，你可以改成 0-15

# ============ 載入模型 ============
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

images = load_and_preprocess_images(image_paths).to(device)
num_frames = len(image_paths)

print(f"\nLoaded {num_frames} frames")

# ============ 提取 Attention ============
result = extract_both_attentions(model, images, block_idx=block_idx)

# 計算 token 資訊
tokens_per_frame = result['frame_attention']['num_tokens']
patch_start_idx = result['patch_start_idx']

print(f"\n========== Token Information ==========")
print(f"Tokens per frame: {tokens_per_frame}")
print(f"Patch start index: {patch_start_idx}")
print(f"Total frames: {num_frames}")

# ============ 計算 Global Attention 中的絕對 token index ============
# Global attention 的 token 排列：[frame0_tokens, frame1_tokens, frame2_tokens, ...]
global_token_idx = frame_idx * tokens_per_frame + target_token_position

print(f"\n========== Your Query ==========")
print(f"Target: Frame {target_frame_number} (index {frame_idx}), Block {target_block_number} (index {block_idx})")
print(f"Token position in frame: {target_token_position}")
print(f"Global token index: {global_token_idx}")

# 檢查 token 是否在有效範圍內
if target_token_position >= tokens_per_frame:
    print(f"\n⚠️  WARNING: Token position {target_token_position} exceeds frame token count {tokens_per_frame}")
    print(f"Valid range: 0 to {tokens_per_frame-1}")
    # 調整到有效範圍
    target_token_position = min(target_token_position, tokens_per_frame - 1)
    global_token_idx = frame_idx * tokens_per_frame + target_token_position
    print(f"Adjusted to: {target_token_position}")

# 如果是 patch token，計算在 grid 中的位置
if target_token_position >= patch_start_idx:
    patch_idx = target_token_position - patch_start_idx
    grid_h = result['frame_attention']['attn_weights'].shape[-1] - patch_start_idx
    grid_h = int(np.sqrt(grid_h))  # 假設是正方形，實際可能是 28×37
    # 從 metadata 獲取實際 grid size
    print(f"Patch index: {patch_idx}")

# ============ 方法1: Frame Attention (該 frame 內部的 attention) ============
print(f"\n========== Extracting Frame Attention ==========")

# Frame attention 需要正確的 batch*frame index
# Frame attention shape: [B*S, num_heads, N, N]
# 對於第 frame_idx 個 frame，我們需要從第 frame_idx 個 batch 中提取
frame_attn_weights = result['frame_attention']['attn_weights'][frame_idx:frame_idx+1]

frame_values = get_attention_values(
    attn_weights=frame_attn_weights,
    images=images[frame_idx:frame_idx+1],  # 只傳該 frame 的圖像
    token_idx=target_token_position,
    head_idx=head_idx,
    patch_start_idx=patch_start_idx,
    attention_type='frame'
)

print(f"Frame attention map shape: {frame_values['attention_maps'].shape}")
print(f"Frame attention resized shape: {frame_values['attention_maps_resized'].shape}")
print(f"Attention value range: [{frame_values['attention_maps'].min():.6f}, {frame_values['attention_maps'].max():.6f}]")

# 儲存數值
np.save(f'frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_frame_attn.npy',
        frame_values['attention_maps_resized'])
print(f"✓ Saved: frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_frame_attn.npy")

# 視覺化 Frame Attention
fig_frame = visualize_attention_on_image(
    attn_weights=frame_attn_weights,
    images=images[frame_idx:frame_idx+1],
    token_idx=target_token_position,
    head_idx=head_idx,
    patch_start_idx=patch_start_idx,
    attention_type='frame'
)

fig_frame.savefig(f'frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_frame_attn.png',
                  dpi=300, bbox_inches='tight')
print(f"✓ Saved: frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_frame_attn.png")

# ============ 方法2: Global Attention (跨所有 frames 的 attention) ============
print(f"\n========== Extracting Global Attention ==========")

global_values = get_attention_values(
    attn_weights=result['global_attention']['attn_weights'],
    images=images,
    token_idx=global_token_idx,
    head_idx=head_idx,
    patch_start_idx=patch_start_idx,
    attention_type='global'
)

print(f"Global attention maps shape: {global_values['attention_maps'].shape}")  # [S, H, W]
print(f"Global attention resized shape: {global_values['attention_maps_resized'].shape}")

# 儲存數值
np.save(f'frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_global_attn.npy',
        global_values['attention_maps_resized'])
print(f"✓ Saved: frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_global_attn.npy")

# 視覺化 Global Attention
fig_global = visualize_attention_on_image(
    attn_weights=result['global_attention']['attn_weights'],
    images=images,
    token_idx=global_token_idx,
    head_idx=head_idx,
    patch_start_idx=patch_start_idx,
    attention_type='global'
)

fig_global.savefig(f'frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_global_attn.png',
                   dpi=300, bbox_inches='tight')
print(f"✓ Saved: frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_global_attn.png")

# ============ 印出 attention values 範例 ============
print(f"\n========== Attention Values Summary ==========")
print(f"Frame Attention (within frame {target_frame_number}):")
print(f"  - Min: {frame_values['attention_maps'].min():.6f}")
print(f"  - Max: {frame_values['attention_maps'].max():.6f}")
print(f"  - Mean: {frame_values['attention_maps'].mean():.6f}")

print(f"\nGlobal Attention (across all {num_frames} frames):")
for i in range(num_frames):
    frame_attn = global_values['attention_maps'][i]
    print(f"  Frame {i+1}: min={frame_attn.min():.6f}, max={frame_attn.max():.6f}, mean={frame_attn.mean():.6f}")

print(f"\n========== Complete! ==========")
print(f"All files saved with prefix: frame_{target_frame_number}_block_{target_block_number}_token_{target_token_position}_")
