import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from spdnet.functions.exponential import SymmetricMatrixExponential
from spdnet.functions.logarithm import SymmetricMatrixLogarithm
from spdnet.functions.power import SymmetricMatrixPower
from spdnet.functions.rectification import SymmetricMatrixRectification


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


@pytest.mark.parametrize("p", [2, -1, 1 / 2])
def test_symmetric_matrix_power(x, p):
    """
    Test the SymmetricMatrixPower function using gradcheck.
    """
    gradcheck_fn(lambda x: SymmetricMatrixPower.apply(x, p), x)
    p = torch.tensor(p, dtype=torch.float64).requires_grad_()
    gradcheck_fn(lambda x: SymmetricMatrixPower.apply(x, p), x)


@pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-6])
def test_symmetric_matrix_rectification(x, eps):
    """
    Test the SymmetricMatrixRectification function using gradcheck.
    """
    gradcheck_fn(lambda x: SymmetricMatrixRectification.apply(x, eps), x)
