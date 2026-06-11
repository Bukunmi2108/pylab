import pytest
import numpy as np, torch
from week01.d1_creation import *

def test_board():
    assert board(3).dtype == torch.float32 and board(3).shape == (3, 3)

def test_dice():
    assert dice(1000, 0).min() >= 1 and dice(1000, 0).max() <= 6

def test_uniform():
    assert torch.equal(uniform((2, 2), 42), uniform((2, 2), 42))

def test_shared_view():
    a = np.zeros(3); t = shared_view(a); a[0] = 5.0
    assert t[0].item() == 5.0

def test_private_copy():
    a2 = np.zeros(3); t2 = private_copy(a2); a2[0] = 5.0
    assert t2[0].item() == 0.0

def test_result_type():
    assert torch.result_type(torch.empty(1, dtype=torch.int64), torch.empty(1, dtype=torch.float32)) == torch.float32