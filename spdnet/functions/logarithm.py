import torch
from torch.autograd import Function

from .bilinear import bilinear
from .linalg import eig2matrix
from .utils import loewner as _loewner
from .utils import symmetrize

__all__ = ["sym_mat_log"]


def sym_mat_log(x: torch.Tensor, svd: bool = True) -> torch.Tensor:
    r"""
    Computes the matrix logarithm of a symmetric positive definite (SPD) matrix using eigendecomposition.

    Given an SPD matrix `x`, this computes:

        log(x) = Q @ diag(log(λ)) @ Q.T

    where `x = Q @ diag(λ) @ Q.T` is the eigendecomposition of `x`,
    and `log(λ)` is applied elementwise.

    Args:
        x (torch.Tensor): Input SPD matrix of shape `(..., N, N)`.
        svd (bool, optional): If True, uses SVD instead of EVD. Defaults to True.

    Returns:
        torch.Tensor: Matrix logarithm of `x`, with shape `(..., N, N)`.
    """
    return SymmetricMatrixLogarithm.apply(x, svd)


class SymmetricMatrixLogarithm(Function):
    r"""
    Autograd-compatible matrix logarithm function for symmetric positive definite (SPD) matrices.

    Forward:
        Computes log(x) by applying log to the eigenvalues of x.

    Backward:
        Uses the Loewner matrix of log to compute the derivative:
            d(log(x)) = V @ [L ⊙ (Vᵀ dY V)] @ Vᵀ,
        where L is the Loewner matrix and ⊙ is element-wise multiplication.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, svd: bool = True) -> torch.Tensor:
        x = symmetrize(x)
        if svd:
            eigvecs, eigvals, _ = torch.linalg.svd(x)
        else:
            eigvals, eigvecs = torch.linalg.eigh(x, "U")

        f_eigvals = torch.log(eigvals)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)

        return symmetrize(eig2matrix(f_eigvals, eigvecs))

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dy: torch.Tensor) -> tuple[torch.Tensor, None]:
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors

        dx = bilinear(dy, eigvecs.mT)
        dx *= loewner(eigvals, f_eigvals)
        dx = bilinear(dx, eigvecs)

        # Ensure symmetry in the gradient
        return symmetrize(dx), None


def loewner(eigvals: torch.Tensor, f_eigvals: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Computes the Loewner matrix for the logarithm function:

        L_{ij} = (log(λ_i) - log(λ_j)) / (λ_i - λ_j),     if i ≠ j
        L_{ii} = 1 / λ_i                                  (derivative of log at λ_i)

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape `(..., N)`.
        f_eigvals (torch.Tensor, optional): Precomputed log(λ). If None, computed internally.

    Returns:
        torch.Tensor: Loewner matrix of shape `(..., N, N)`.
    """
    if f_eigvals is None:
        f_eigvals = torch.log(eigvals)

    eps = torch.finfo(eigvals.dtype).eps
    df_eigvals = 1.0 / (eigvals + eps)  # Avoid division by zero for very small eigenvalues

    return _loewner(eigvals, f_eigvals, df_eigvals)
