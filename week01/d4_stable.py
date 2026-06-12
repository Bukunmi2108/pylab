"""Week One - Day 4"""

import torch


def naive_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return ( torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True))

def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.amax(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_val)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.amax(x, dim=dim, keepdim=True)
    return (x - max_val) - (x - max_val).exp().sum(dim, keepdim=True).log()

def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.amax(x, dim=dim, keepdim=True)
    out = (x - max_val).exp().sum(dim, keepdim=True).log() + max_val
    return out.squeeze(dim)

def cross_entropy_from_scratch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_val = torch.amax(logits, dim=-1, keepdim=True)
    logsumexp = (logits - max_val).exp().sum(dim=-1, keepdim=True).log() + max_val
    logsumexp = logsumexp.squeeze(-1)
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    target_logits = logits[batch_indices, targets]    
    loss = logsumexp - target_logits
    return loss.mean()
