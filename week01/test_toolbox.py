import torch
import pytest
from week01.toolbox import pairwise_dist, one_hot, make_blobs, knn_predict

@pytest.fixture
def device():
    """Dynamically provides the active execution device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_pairwise_dist(device):
    a = torch.randn(50, 8, device=device)
    b = torch.randn(30, 8, device=device)
    
    res = pairwise_dist(a, b)
    ref = torch.cdist(a, b)
    assert torch.allclose(res, ref, atol=1e-4)

def test_one_hot(device):
    labels = torch.tensor([0, 2, 1], dtype=torch.int64, device=device)
    expected = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ], dtype=torch.float32, device=device)
    assert torch.equal(one_hot(labels, num_classes=3), expected)

def test_make_blobs_and_knn_accuracy(device):
    centers = torch.tensor([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]], device=device)
    X, y = make_blobs(n_per_class=60, centers=centers, std=1.0, seed=0)
    train_x, train_y = X[::2], y[::2]
    test_x, test_y = X[1::2], y[1::2]
    predictions = knn_predict(train_x, train_y, test_x, k=5)
    accuracy = (predictions == test_y).float().mean().item()
    assert accuracy > 0.9, f"Expected accuracy > 0.9, got {accuracy:.4f}"
