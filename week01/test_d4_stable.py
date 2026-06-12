import torch
import torch.nn.functional as F
import pytest
from week01.d4_stable import *


@pytest.fixture
def sample_logits():
    """Fixture to provide a standard random batch of logits."""
    return torch.randn(8, 10)


def test_stable_softmax(sample_logits):
    """Verify stable_softmax matches PyTorch functional softmax."""
    assert torch.allclose(
        stable_softmax(sample_logits), 
        F.softmax(sample_logits, dim=-1), 
        atol=1e-6
    )


def test_stable_log_softmax(sample_logits):
    """Verify stable_log_softmax matches PyTorch functional log_softmax."""
    assert torch.allclose(
        stable_log_softmax(sample_logits), 
        F.log_softmax(sample_logits, dim=-1), 
        atol=1e-6
    )


def test_logsumexp(sample_logits):
    """Verify custom logsumexp matches PyTorch native logsumexp."""
    assert torch.allclose(
        logsumexp(sample_logits), 
        torch.logsumexp(sample_logits, dim=-1), 
        atol=1e-6
    )


def test_numerical_extremes():
    """Verify operations remain stable or fail as expected with massive inputs."""
    big = torch.tensor([[1e4, -1e4, 0.0]])
    assert torch.isnan(naive_softmax(big)).any()
    
    assert torch.allclose(stable_softmax(big), F.softmax(big, dim=-1))
    assert torch.isfinite(stable_log_softmax(big)).all()


def test_cross_entropy_accuracy(sample_logits):
    """Verify scratch cross entropy matches PyTorch functional cross entropy."""
    targets = torch.randint(0, 10, (8,))
    assert torch.allclose(
        cross_entropy_from_scratch(sample_logits, targets), 
        F.cross_entropy(sample_logits, targets), 
        atol=1e-6
    )


def test_cross_entropy_uniform_distribution():
    """Verify uniform logits over K classes mathematically equal ln(K)."""
    K = 7
    uniform_logits = torch.zeros(4, K)
    random_targets = torch.randint(0, K, (4,))
    expected_loss = torch.tensor(K, dtype=torch.float32).log()
    
    assert torch.allclose(
        cross_entropy_from_scratch(uniform_logits, random_targets), 
        expected_loss
    )
