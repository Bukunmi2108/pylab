"""Week One - Day Six"""

import torch

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    n = labels.size(0)
    out = torch.zeros((n, num_classes), dtype=torch.float32, device=labels.device)
    out[torch.arange(n), labels] = 1.0
    return out

def moving_average(x: torch.Tensor, w: int) -> torch.Tensor:
    c = torch.cumsum(x, dim=0)
    c = torch.cat([torch.zeros(1, dtype=c.dtype, device=c.device), c])
    return (c[w:] - c[:-w]) / w

def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sq_norm_a = torch.sum(a ** 2, dim=1, keepdim=True)
    sq_norm_b = torch.sum(b ** 2, dim=1, keepdim=True)
    cross_term = torch.matmul(a, b.T)
    sq_dist = sq_norm_a + sq_norm_b.T - 2 * cross_term
    return torch.sqrt(torch.clamp(sq_dist, min=0.0))

def batched_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., :, None] * b[..., None, :]

def make_blobs(n_per_class: int, centers: torch.Tensor, std: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=centers.device).manual_seed(seed)
    K, d = centers.shape
    expanded_centers = centers.unsqueeze(1).expand(-1, n_per_class, -1).reshape(-1, d)
    noise = torch.randn(K * n_per_class, d, generator=generator, device=centers.device) * std
    labels = torch.arange(K, dtype=torch.int64, device=centers.device).unsqueeze(1).expand(-1, n_per_class).reshape(-1)
    return expanded_centers + noise, labels

def knn_predict(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, k: int = 5) -> torch.Tensor:
    dists = pairwise_dist(test_x, train_x)
    _, indices = torch.topk(dists, k=k, dim=1, largest=False)
    neighbor_labels = train_y[indices]
    predictions, _ = torch.mode(neighbor_labels, dim=1)
    return predictions
