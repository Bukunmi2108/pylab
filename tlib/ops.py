import torch

def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.amax(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_val)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.amax(x, dim=dim, keepdim=True)
    return (x - max_val) - (x - max_val).exp().sum(dim, keepdim=True).log()

def cross_entropy_from_scratch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_val = torch.amax(logits, dim=-1, keepdim=True)
    logsumexp = (logits - max_val).exp().sum(dim=-1, keepdim=True).log() + max_val
    logsumexp = logsumexp.squeeze(-1)
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    target_logits = logits[batch_indices, targets]    
    loss = logsumexp - target_logits
    return loss.mean()

def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sq_norm_a = torch.sum(a ** 2, dim=1, keepdim=True)
    sq_norm_b = torch.sum(b ** 2, dim=1, keepdim=True)
    cross_term = torch.matmul(a, b.T)
    sq_dist = sq_norm_a + sq_norm_b.T - 2 * cross_term
    return torch.sqrt(torch.clamp(sq_dist, min=0.0))

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    n = labels.size(0)
    out = torch.zeros((n, num_classes), dtype=torch.float32, device=labels.device)
    out[torch.arange(n), labels] = 1.0
    return out