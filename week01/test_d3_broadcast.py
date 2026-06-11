import pytest
import random
import torch
from week01.d3_broadcast import *


def test_randomized_broadcasting():
    """Runs 200 randomized compatibility checks against PyTorch broadcasting."""
    allowed_dims = [1, 2, 3, 5]
    iterations = 200
    
    for _ in range(iterations):
        rank_a = random.randint(0, 4)
        rank_b = random.randint(0, 4)
        
        shape_a = tuple(random.choice(allowed_dims) for _ in range(rank_a))
        shape_b = tuple(random.choice(allowed_dims) for _ in range(rank_b))
        
        try:
            expected_shape = torch.broadcast_shapes(shape_a, shape_b)
            torch_failed = False
        except RuntimeError:
            torch_failed = True
            
        if torch_failed:
            with pytest.raises(ValueError):
                broadcast_shape(shape_a, shape_b)
        else:
            actual_shape = broadcast_shape(shape_a, shape_b)
            assert actual_shape == expected_shape, f"Mismatch! a:{shape_a} b:{shape_b}"
    
def test_broadcast_shape_standard():
    assert broadcast_shape((5, 1, 3), (4, 3)) == (5, 4, 3)


def test_broadcast_shape_scalar():
    assert broadcast_shape((), (3, 2)) == (3, 2)


def test_broadcast_shape_incompatible():
    with pytest.raises(ValueError):
        broadcast_shape((3, 2), (2, 2))


def test_pairwise_diff():
    a, b = torch.randn(4, 3), torch.randn(5, 3)
    assert pairwise_diff(a, b).shape == (4, 5, 3)
    assert torch.allclose(pairwise_diff(a, b)[2, 3], a[2] - b[3])


def test_outer_product():
    v1 = torch.arange(3.)
    v2 = torch.arange(4.)
    assert torch.allclose(outer(v1, v2), torch.outer(v1, v2))


def test_de_mean_cols():
    m = torch.randn(6, 4)
    assert torch.allclose(de_mean_cols(m).mean(dim=0), torch.zeros(4), atol=1e-6)


def test_normalize_rows():
    m = torch.randn(5, 3)
    assert torch.allclose(normalize_rows(m).norm(dim=1), torch.ones(5), atol=1e-6)
