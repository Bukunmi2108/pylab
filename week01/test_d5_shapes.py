import pytest
import torch
from week01.d5_shapes import *

@pytest.fixture
def base_tensor():
    return torch.randn(2, 4, 6, 3)


# --- Test Cases ---
def test_nhwc_to_nchw_conversions(base_tensor):
    x = base_tensor
    assert nhwc_to_nchw(x).shape == (2, 3, 4, 6)
    assert torch.equal(nchw_to_nhwc(nhwc_to_nchw(x)), x)
    assert nhwc_to_nchw(x)[0, 1, 2, 3] == x[0, 2, 3, 1]


def test_to_patches():
    img = torch.arange(2 * 4 * 4.).reshape(2, 4, 4)
    P = to_patches(img, 2)
    assert P.shape == (4, 8)
    expected_patch = torch.stack([img[0, :2, :2], img[1, :2, :2]]).reshape(-1)
    assert torch.equal(P[0], expected_patch)


def test_attention_heads():
    h = split_heads(torch.randn(2, 5, 12), 3)
    assert h.shape == (2, 3, 5, 4)
    seq = torch.arange(48.).reshape(1, 4, 12)
    assert torch.equal(merge_heads(split_heads(seq, 4)), seq)


def test_broadcast_row_stride():
    assert broadcast_row(torch.arange(3.), 4).stride(0) == 0