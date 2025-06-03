import pytest
from spdnet.functions.autograd import *
from torch.autograd import gradcheck, gradgradcheck

import torch
from ..toy_data import make_spd_matrix

torch.manual_seed(0)


@pytest.fixture(scope="module")
def x():
    """
    Fixture to create a symmetric positive definite matrix for testing.
    """
    batch = 5

    return torch.stack([make_spd_matrix(4, dtype=torch.float64).requires_grad_() for _ in range(batch)], dim=0)


def gradcheck_fn(func, x):
    """
    Helper function to perform gradcheck on a given function with input x.
    """
    # Batched input testing
    assert gradcheck(func, [x])
    # Single input testing
    assert gradcheck(func, [x[0].unsqueeze(0)])


def test_symmetric_matrix_logarithm(x):
    """
    Test the SymmetricMatrixLogarithm function using gradcheck.
    """
    gradcheck_fn(SymmetricMatrixLogarithm.apply, x)


def test_symmetric_matrix_exponential(x):
    """
    Test the SymmetricMatrixExponential function using gradcheck.
    """
    gradcheck_fn(SymmetricMatrixExponential.apply, x)


def test_symmetric_matrix_power(x):
    """
    Test the SymmetricMatrixPower function using gradcheck.
    """
    for p in [2, -1, 1 / 2]:
        gradcheck_fn(lambda x: SymmetricMatrixPower.apply(x, p), x)


def test_symmetric_matrix_rectification(x):
    """
    Test the SymmetricMatrixRectification function using gradcheck.
    """
    eps = 1e-5
    gradcheck_fn(lambda x: SymmetricMatrixRectification.apply(x, eps), x)
