"""Week One - Day Two - Indexing"""

import torch

def anti_diagonal(m: torch.Tensor) -> torch.Tensor:
    n = m.size(0)
    return m[torch.arange(n), (n - 1) - torch.arange(n)]

def select_per_row(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return m[torch.arange(m.size(0)), idx]

def reverse_every_other_row(m: torch.Tensor) -> torch.Tensor:
    m[::2] = torch.flip(m[::2], [1])
    return m

def positives(x: torch.Tensor) -> torch.Tensor:
    return x[x > 0]

def relu_(x: torch.Tensor) -> torch.Tensor:
    x[x < 0] = 0
    return x

def border(m: torch.Tensor) -> torch.Tensor:
    m[1:-1, 1:-1] = 0
    return m

def last_col(t: torch.Tensor) -> torch.Tensor:
    return t[..., -1]

def as_column(x: torch.Tensor) -> torch.Tensor:
    return x[:, None]

def swap_halves(x: torch.Tensor) -> torch.Tensor:
    half1, half2 = x.chunk(2)
    return torch.cat((half2, half1))

def checkerboard(m: torch.Tensor) -> torch.Tensor:
    out = m.clone()
    out[0::2, 1::2] = 0
    out[1::2, 0::2] = 0
    return out
