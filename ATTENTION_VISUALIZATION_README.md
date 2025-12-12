# VGGT Attention Visualization Guide

這個工具可以提取並可視化 VGGT 模型的 attention weights，支援 **Frame Attention** 和 **Global Attention**。

## 功能特色

### 1. **在實際影像上可視化 Attention**
- 將 attention weights 疊加在原始影像上
- 可以看到某個 token 關注影像的哪些區域
- 支援切換不同的 token 進行可視化

### 2. **同時提取 Frame 和 Global Attention**
- Frame Attention：幀內的 self-attention
- Global Attention：跨幀的 cross-frame attention，會分別顯示每一幀

### 3. **提取純數值 Attention Maps**
- 新增 `get_attention_values()` 函數
- 返回 numpy arrays 而非圖片，方便後續處理
- 可自行儲存、分析或客製化可視化

### 4. **靈活的參數設定**
- 選擇任意 block (0-23)
- 選擇任意 attention head (0-15)
- 選擇任意 token 進行可視化

## 快速開始

### 基本使用

```python
from visualize_attention import demo_visualize_token_attention

image_paths = [
    "path/to/image1.png",
    "path/to/image2.png",
    "path/to/image3.png"
]

# 可視化 token 500 的 attention
figures, result = demo_visualize_token_attention(
    image_paths=image_paths,
    block_idx=23,        # 最後一層
    token_idx=500,       # 要可視化的 token
    head_idx=0,          # attention head
    attention_types=['frame', 'global']  # 兩種都顯示
)
```

### 進階使用：手動控制

```python
from visualize_attention import (
    AttentionExtractor,
    extract_both_attentions,
    visualize_attention_on_image
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import torch

# 載入模型和影像
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

images = load_and_preprocess_images(image_paths).to(device)

# 提取 attention
result = extract_both_attentions(model, images, block_idx=23)

# 可視化特定 token
fig = visualize_attention_on_image(
    attn_weights=result['global_attention']['attn_weights'],
    images=images,
    token_idx=500,
    head_idx=0,
    patch_start_idx=result['patch_start_idx'],
    attention_type='global'
)
```

## 核心函數說明

### `demo_visualize_token_attention()`
最簡單的使用方式，一鍵完成所有操作。

**參數：**
- `image_paths`: 影像路徑列表
- `block_idx`: Transformer block 索引 (0-23)
- `token_idx`: 要可視化的 token 索引
- `head_idx`: Attention head 索引 (0-15)
- `attention_types`: `['frame', 'global']` 或其中之一

**返回：**
- `figures`: matplotlib Figure 物件列表
- `result`: 包含 attention weights 的字典

### `extract_both_attentions()`
提取指定 block 的 frame 和 global attention。

**返回字典結構：**
```python
{
    'frame_attention': {
        'attn_weights': torch.Tensor,  # [B*S, num_heads, N, N]
        'layer_type': 'frame',
        'num_heads': 16,
        'num_tokens': N
    },
    'global_attention': {
        'attn_weights': torch.Tensor,  # [B, num_heads, S*P, S*P]
        'layer_type': 'global',
        'num_heads': 16,
        'num_tokens': S*P
    },
    'block_idx': 23,
    'patch_start_idx': 5
}
```

### `visualize_attention_on_image()`
在實際影像上可視化 attention weights。

**Frame Attention 輸出：**
- 左圖：原始影像
- 右圖：Attention heatmap 疊加在影像上

**Global Attention 輸出：**
- 上排：所有幀的原始影像
- 下排：每一幀的 attention heatmap

### `get_attention_values()`
提取純數值的 attention weights，不做可視化。與 `visualize_attention_on_image()` 參數完全相同，但返回 numpy arrays 而非圖片。

**用途：**
- 需要自己處理 attention 數據
- 要儲存 attention maps 做後續分析
- 想要客製化可視化方式

**參數：**（與 `visualize_attention_on_image()` 相同）
- `attn_weights`: Attention weights tensor
- `images`: 原始影像 tensor
- `token_idx`: 要提取的 token 索引
- `head_idx`: Attention head 索引
- `patch_start_idx`: Patch tokens 起始索引
- `attention_type`: 'frame' 或 'global'

**返回字典結構：**
```python
{
    'attention_maps': np.ndarray,
        # Frame: [grid_h, grid_w] - 原始 patch 解析度
        # Global: [S, grid_h, grid_w] - 每一幀的 attention map

    'attention_maps_resized': np.ndarray,
        # Frame: [H, W] - 放大到影像解析度
        # Global: [S, H, W] - 每一幀放大後的 attention map

    'images': np.ndarray,  # [S, H, W, 3] - 原始影像，range [0, 1]

    'metadata': {
        'grid_h': int,           # patch grid 高度
        'grid_w': int,           # patch grid 寬度
        'H': int,                # 影像高度
        'W': int,                # 影像寬度
        'S': int,                # 影像數量
        'token_idx': int,        # 查詢的 token
        'head_idx': int,         # attention head
        'attention_type': str,   # 'frame' or 'global'
        'patch_start_idx': int   # patch tokens 起始索引
    }
}
```

**使用範例：**
```python
from visualize_attention import get_attention_values

# 提取 attention
result = extract_both_attentions(model, images, block_idx=23)

# 取得純數值（不做可視化）
values = get_attention_values(
    attn_weights=result['global_attention']['attn_weights'],
    images=images,
    token_idx=500,
    head_idx=0,
    patch_start_idx=result['patch_start_idx'],
    attention_type='global'
)

# Frame Attention
frame_values = get_attention_values(
    attn_weights=result['frame_attention']['attn_weights'],
    images=images,
    token_idx=500,
    head_idx=0,
    patch_start_idx=result['patch_start_idx'],
    attention_type='frame'
)

# 取得 attention map
attn_map = frame_values['attention_maps']  # [37, 37]
attn_resized = frame_values['attention_maps_resized']  # [518, 518]

print(f"Attention map shape: {attn_map.shape}")
print(f"Attention range: [{attn_resized.min():.4f}, {attn_resized.max():.4f}]")

# Global Attention - 每一幀的 attention
global_values = get_attention_values(
    attn_weights=result['global_attention']['attn_weights'],
    images=images,
    token_idx=500,
    head_idx=0,
    patch_start_idx=result['patch_start_idx'],
    attention_type='global'
)

attn_maps = global_values['attention_maps']  # [3, 37, 37]
attn_resized = global_values['attention_maps_resized']  # [3, 518, 518]

# 自己客製化可視化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axes[i].imshow(attn_resized[i], cmap='turbo')
    axes[i].set_title(f'Frame {i}')
    axes[i].axis('off')
plt.savefig('custom_attention.png')

# 儲存成 .npy 檔案
import numpy as np
np.save('attention_maps.npy', attn_resized)
```

**與 `visualize_attention_on_image()` 的比較：**

| 特性 | `visualize_attention_on_image()` | `get_attention_values()` |
|------|----------------------------------|--------------------------|
| 返回值 | matplotlib Figure | numpy arrays (dict) |
| 用途 | 直接可視化並顯示 | 取得數值做後續處理 |
| 輸出類型 | 圖片物件 | 純數據 + metadata |
| 彈性 | 低（固定格式） | 高（可自由處理） |
| 適用場景 | 快速查看結果 | 需要分析、儲存或客製化 |

## Token 索引說明

VGGT 的 token 結構：
```
[camera_token, register_tokens (4個), patch_tokens...]
```

- **Special tokens**: 索引 0-4 (camera + register tokens)
- **Patch tokens**: 從索引 5 開始
- 對於 518x518 影像，patch grid 是 37x37 = 1369 個 patches
- **Valid token range**: 5 到 1373

### 如何選擇 token_idx？

Token 的空間位置對應：
```python
# 假設 grid 是 37x37
patch_start_idx = 5
grid_h, grid_w = 37, 37

# Token idx 轉空間位置
token_idx = 500
patch_idx = token_idx - patch_start_idx  # 495
row = patch_idx // grid_w  # 13 (從上數第14排)
col = patch_idx % grid_w   # 12 (從左數第13個)

# 空間位置轉 token idx
row, col = 18, 18  # 影像中心
patch_idx = row * grid_w + col
token_idx = patch_idx + patch_start_idx
```

**常用 token 位置：**
- 左上角：`token_idx = 5`
- 影像中心：`token_idx = 5 + (grid_h//2 * grid_w + grid_w//2)`
- 右下角：`token_idx = 5 + grid_h * grid_w - 1`

## Attention 類型差異

### Frame Attention
- **Shape**: `[B*S, num_heads, N, N]` where N = patches per frame + special tokens
- **含義**: 每一幀內部的 self-attention
- **用途**: 理解單張影像內的空間關係

### Global Attention
- **Shape**: `[B, num_heads, S*P, S*P]` where S = frames, P = patches per frame
- **含義**: 所有幀之間的 cross-frame attention
- **用途**: 理解不同影像之間的對應關係（例如：tracking, 3D reconstruction）

## 可視化範例

### Example 1: 查看影像中心的 attention
```python
# 計算中心 token
grid_size = 37
center_patch = (grid_size // 2) * grid_size + (grid_size // 2)
center_token = center_patch + 5  # 加上 special tokens

demo_visualize_token_attention(
    image_paths=image_paths,
    block_idx=23,
    token_idx=center_token,
    head_idx=0
)
```

### Example 2: 比較不同 layer 的 attention
```python
for block_idx in [0, 12, 23]:  # Early, middle, deep
    demo_visualize_token_attention(
        image_paths=image_paths,
        block_idx=block_idx,
        token_idx=500,
        head_idx=0,
        attention_types=['global']
    )
```

### Example 3: 比較不同 attention head
```python
for head_idx in range(16):
    demo_visualize_token_attention(
        image_paths=image_paths,
        block_idx=23,
        token_idx=500,
        head_idx=head_idx,
        attention_types=['frame']
    )
```

## 技術細節

### Hook 機制
使用 PyTorch 的 `register_forward_hook` 來攔截 forward pass：

```python
def hook_fn(module, input, output):
    # 手動計算 attention weights
    x = input[0]
    q, k, v = compute_qkv(x)
    attn_weights = softmax(q @ k.T)
    # 儲存結果
    self.attention_maps[layer_name] = attn_weights
```

### 為什麼需要手動計算？
VGGT 使用 **fused attention** 來優化效能，這種實作不會返回中間的 attention weights。因此我們需要：
1. 用 hook 攔截輸入
2. 手動重新計算 Q, K
3. 計算 `softmax(QK^T)` 得到 attention map

### Global Attention 的 Frame 分離
Global attention 的 shape 是 `[B, num_heads, S*P, S*P]`，其中 tokens 按以下順序排列：
```
[frame0_patch0, frame0_patch1, ..., frame1_patch0, frame1_patch1, ...]
```

要分離每一幀的 attention：
```python
attn = attn_weights[0, head_idx, token_idx, patch_start_idx:]
attn_per_frame = attn.reshape(S, num_patches)  # [S, P]
```

## 常見問題

### Q: 為什麼我的 token_idx 被自動調整了？
A: token_idx 必須在有效範圍內（>= patch_start_idx 且 < patch_start_idx + num_patches）。如果超出範圍，會自動調整到最近的有效值。

### Q: 為什麼 attention map 看起來很模糊？
A: Attention map 的原始解析度是 37x37（對應 patch grid），我們使用 bilinear interpolation 放大到原始影像尺寸。

### Q: 如何儲存結果？
A:
```python
figures, result = demo_visualize_token_attention(...)
figures[0][1].savefig('frame_attention.png', dpi=300, bbox_inches='tight')
figures[1][1].savefig('global_attention.png', dpi=300, bbox_inches='tight')
```

### Q: 可以提取多個 block 的 attention 嗎？
A: 可以！
```python
extractor = AttentionExtractor(model)
extractor.register_hooks(
    block_indices=[0, 12, 23],  # 多個 blocks
    attention_types=['frame', 'global']
)
# Forward pass
with torch.no_grad():
    _ = model(images)
attention_maps = extractor.get_attention_maps()
```

## 效能建議

1. **記憶體使用**：Attention maps 很大，建議一次只處理一個 block
2. **計算開銷**：手動計算 attention 會增加約 20-30% 的計算時間
3. **清理 hooks**：使用完畢後記得呼叫 `extractor.clear_hooks()` 來移除 hooks

## 參考資料

- VGGT Paper: [Visual Geometry Grounded Transformer]
- PyTorch Hooks: [torch.nn.Module.register_forward_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.register_forward_hook.html)
