import pytest
import torch
from torch.autograd import gradcheck

from spdnet.functions import bilinear, sym_mat_exp, sym_mat_log, sym_mat_pow, sym_mat_rec


def gradcheck_fn(func, x: torch.Tensor, *args: torch.Tensor) -> None:
    r"""
    Helper function to perform gradcheck on both batched and single samples.

    Args:
        func (callable): Function to test.
        x (torch.Tensor): Input tensor of shape (..., N, N).
        *args (torch.Tensor): Additional arguments to `func`.
    """
    # Batched input
    assert gradcheck(func, [x.requires_grad_(), *args])

    # Single instance input
    x0 = x[0].detach().clone().requires_grad_()
    args0 = [arg[0] if isinstance(arg, torch.Tensor) and arg.shape == x.shape else arg for arg in args]
    assert gradcheck(func, [x0, *args0])


def test_bilinear(x: torch.Tensor, w: torch.Tensor) -> None:
    r"""
    Test the `bilinear` function with batched and singleton inputs.
    """
    gradcheck_fn(bilinear, x, w)


@pytest.mark.parametrize("svd", [True, False])
def test_sym_mat_log(x: torch.Tensor, svd: bool) -> None:
    r"""
    Test the matrix logarithm function `sym_mat_log`.
    """
    gradcheck_fn(sym_mat_log, x, svd)


def test_sym_mat_exp(x: torch.Tensor) -> None:
    r"""
    Test the matrix exponential function `sym_mat_exp`.

    Applies `log` first to ensure numerically stable SPD input.
    """
    x = sym_mat_log(x)  # ensures stable spectrum for exp
    gradcheck_fn(sym_mat_exp, x)


@pytest.mark.parametrize("svd", [True, False])
@pytest.mark.parametrize("p", [0.0, 2.0, -1.0, 0.5])
def test_sym_mat_pow(x: torch.Tensor, p: float, svd: bool) -> None:
    r"""
    Test matrix power function `sym_mat_pow` for fixed and differentiable exponents.

    Args:
        x (torch.Tensor): SPD matrix input.
        p (float): Scalar power to test.
        svd (bool): Whether to use SVD or EVD for computation.
    """
    # Static exponent
    gradcheck_fn(sym_mat_pow, x, p, svd)

    # Learnable exponent
    p_tensor = torch.tensor(p, dtype=x.dtype, requires_grad=True)
    gradcheck_fn(sym_mat_pow, x, p_tensor)

    # Optional: second-order test
    # assert gradgradcheck(lambda x: sym_mat_pow(x, p_tensor), [x])


@pytest.mark.parametrize("svd", [True, False])
@pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-6])
def test_sym_mat_rec(x: torch.Tensor, eps: float, svd: bool) -> None:
    r"""
    Test SPD rectification function `sym_mat_rec` with different clamping thresholds.
    """
    gradcheck_fn(sym_mat_rec, x, eps, svd)
