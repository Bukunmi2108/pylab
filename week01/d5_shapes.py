"""Week One - Day Five"""

import torch

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return torch.permute(x, (0,3,1,2))

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return torch.permute(x, (0,2,3,1))

def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1)

def to_patches(img: torch.Tensor, p: int) -> torch.Tensor:
    C, H, W = img.shape
    x = img.reshape(C, H // p, p, W // p, p)
    x = x.permute(1, 3, 0, 2, 4)
    return x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, T, D = x.shape
    d_k = D // n_heads
    x = x.reshape(B, T, n_heads, d_k)
    return x.transpose(1, 2)

def merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, n_heads, T, d_k = x.shape
    D = n_heads * d_k
    x = x.transpose(1, 2)
    return x.reshape(B, T, D)

def drop_unit_dims(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(-1)

def broadcast_row(row: torch.Tensor, n: int) -> torch.Tensor:
    return row.unsqueeze(0).expand(n, -1)

def tile_row(row: torch.Tensor, n: int) -> torch.Tensor:
    return row.repeat(n)