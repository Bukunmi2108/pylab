import torch
import pytest
from week01.d2_indexing import (
    anti_diagonal,
    select_per_row,
    reverse_every_other_row,
    relu_,
    last_col,
    as_column
)

@pytest.fixture
def base_matrix():
    """Provides a fresh 3x3 matrix for indexing tests."""
    return torch.arange(9).reshape(3, 3)


def test_anti_diagonal(base_matrix):
    expected = torch.tensor([2, 4, 6])
    assert torch.equal(anti_diagonal(base_matrix), expected)


def test_select_per_row(base_matrix):
    indices = torch.tensor([2, 0, 1])
    expected = torch.tensor([2, 3, 7])
    assert torch.equal(select_per_row(base_matrix, indices), expected)


def test_reverse_every_other_row(base_matrix):
    result = reverse_every_other_row(base_matrix)
    assert torch.equal(result[0], torch.tensor([2, 1, 0]))
    assert torch.equal(result[1], torch.tensor([3, 4, 5]))


def test_relu_inplace():
    x = torch.tensor([-1.0, 2.0, -3.0])
    relu_(x)
    assert torch.equal(x, torch.tensor([0.0, 2.0, 0.0]))


def test_last_col_shape():
    m = torch.zeros(2, 3, 4)
    assert last_col(m).shape == (2, 3)


def test_as_column_shape():
    m = torch.arange(5)
    assert as_column(m).shape == (5, 1)
