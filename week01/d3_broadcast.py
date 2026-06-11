"""Week One - Day 3"""

import torch

def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    n = len(a)
    m = len(b)

    if a == b: return a

    def merge(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
        result = []
        for dima, dimb in zip(a, b):
            if dima == dimb:
                result.append(dima)
            elif dima == 1:
                result.append(dimb)
            elif dimb == 1:
                result.append(dima)
            else:
                raise ValueError(f"incompatible: {a} vs {b}")
        return tuple(result)
    
    if n == m:
        return merge(a, b)
    else:
        max_len = max(n, m)
        padded_a = (1,) * (max_len - n) + a
        padded_b = (1,) * (max_len - m) + b
        return merge(padded_a, padded_b)
    

def de_mean_cols(m: torch.Tensor) -> torch.Tensor:
    return m - m.mean(dim=0)

def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, None] * b[None, :]

def pairwise_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, None, :] - b[None, :, :]

def normalize_rows(m: torch.Tensor) -> torch.Tensor:
    return m / torch.norm(m, p=2, dim=1, keepdim=True)

def dist_from(points: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((points - q) ** 2, dim=1))

def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return x + b
