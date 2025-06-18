import torch
from torch.autograd import Function

from .bilinear import bilinear
from .linalg import eig2matrix
from .utils import loewner as _loewner

__all__ = ["sym_mat_exp"]


def sym_mat_exp(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the matrix exponential of a symmetric matrix using eigendecomposition.

    For a symmetric matrix `x`, this computes:

        exp(x) = Q @ diag(exp(λ)) @ Q.T

    where `x = Q @ diag(λ) @ Q.T` is the eigendecomposition of `x`.

    Args:
        x (torch.Tensor): Symmetric matrix of shape `(..., N, N)`.

    Returns:
        torch.Tensor: Matrix exponential of shape `(..., N, N)`.
    """
    return SymmetricMatrixExponential.apply(x)


class SymmetricMatrixExponential(Function):
    r"""
    Autograd-compatible function for computing the matrix exponential of symmetric matrices.

    Forward:
        Computes `exp(x)` via eigendecomposition and applies the exponential to eigenvalues.

    Backward:
        Uses the Loewner matrix to compute the derivative of `exp(x)`:
            d(exp(x)) = V @ [L ⊙ (Vᵀ dY V)] @ Vᵀ,
        where ⊙ denotes element-wise product and L is the Loewner matrix.

    This method is numerically stable and preserves symmetry.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.exp(eigvals)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)

        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dy: torch.Tensor) -> tuple[torch.Tensor]:
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors

        # Transform dy to eigenbasis
        dx = bilinear(dy, eigvecs.mT)

        # Multiply by Loewner matrix for exp
        dx *= loewner(eigvals, f_eigvals)

        # Project back
        dx = bilinear(dx, eigvecs)

        # Ensure symmetry in gradient
        return (dx + dx.mT) / 2


def loewner(eigvals: torch.Tensor, f_eigvals: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Computes the Loewner matrix for the exponential function:
        L_{ij} = (exp(λ_i) - exp(λ_j)) / (λ_i - λ_j)
    and
        L_{ii} = exp(λ_i)

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape `(..., N)`.
        f_eigvals (torch.Tensor, optional): Precomputed exp(eigvals). If None, computed internally.

    Returns:
        torch.Tensor: Loewner matrix of shape `(..., N, N)`.
    """
    if f_eigvals is None:
        f_eigvals = torch.exp(eigvals)

    # Since f = exp, df = exp(λ), so use f_eigvals for both numerator and diagonal derivative
    return _loewner(eigvals, f_eigvals, f_eigvals)
