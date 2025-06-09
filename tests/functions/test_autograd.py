import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from spdnet.functions import bilinear, sym_mat_exp, sym_mat_log, sym_mat_pow, sym_mat_rec


def gradcheck_fn(func, x, *args):
    """
    Helper function to perform gradcheck on a given function with input x.
    """
    # Batched input testing
    assert gradcheck(func, [x, *args])
    # Single input testing
    assert gradcheck(func, [x[0], *args])


def test_bilinear(x, w):
    """
    Test the bilinear function using gradcheck.
    """
    gradcheck_fn(bilinear, x, w)


def test_sym_mat_log(x):
    """
    Test the sym_mat_log function using gradcheck.
    """
    gradcheck_fn(sym_mat_log, x)


def test_sym_mat_exp(x):
    """
    Test the sym_mat_exp function using gradcheck.
    """
    # Ensure stable input for exp
    x = sym_mat_log(x)
    gradcheck_fn(sym_mat_exp, x)


@pytest.mark.parametrize("p", [2, -1, 1 / 2])
def test_sym_mat_pow(x, p):
    """
    Test the sym_mat_pow function using gradcheck.
    """
    gradcheck_fn(lambda x: sym_mat_pow(x, p), x)
    p = torch.tensor(p, dtype=torch.float64).requires_grad_()
    gradcheck_fn(lambda x: sym_mat_pow(x, p), x)


@pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-6])
def test_sym_mat_rec(x, eps):
    """
    Test the sym_mat_rec function using gradcheck.
    """
    gradcheck_fn(lambda x: sym_mat_rec(x, eps), x)
