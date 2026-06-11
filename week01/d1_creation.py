"""Week One, Day One"""

import torch
import numpy as np

def board(n: int) -> torch.Tensor:
    return torch.zeros(n, n, dtype=torch.float32)

def ones_long(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.int64)

def seven(rows: int, col:int) -> torch.Tensor:
    return torch.full((rows, col), 7, dtype=torch.float32)

def identity_f64(n: int) -> torch.Tensor:
    return torch.eye(n, dtype=torch.float64)

def evens_below(n: int) -> torch.Tensor:
    return torch.arange(0, n, step=2, dtype=torch.int64)

def grid(start: float, stop: float, num: int) -> torch.Tensor:
    return torch.linspace(start, stop, num)

def uniform(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(shape)

def gaussian(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(shape)

def dice(n: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(1, 7, (n,))

def from_list32(data: list) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)

def shared_view(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr)

def private_copy(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr)
