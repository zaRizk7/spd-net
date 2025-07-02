import torch
from torch.autograd import Function

from .bilinear import bilinear
from .linalg import eig2matrix
from .utils import loewner as _loewner
from .utils import symmetrize

__all__ = ["sym_mat_rec"]


def sym_mat_rec(x: torch.Tensor, eps: float = 1e-5, svd: bool = True) -> torch.Tensor:
    r"""
    Rectifies a symmetric matrix by clamping its eigenvalues to a minimum threshold.

    Given a symmetric matrix `x`, this computes:

        rect(x) = Q @ diag(clamp(λ, min=eps)) @ Q.T

    where `x = Q @ diag(λ) @ Q.T` is the eigendecomposition of `x`.

    This operation ensures the output remains symmetric positive definite (SPD),
    and is commonly used to stabilize matrices that are nearly singular.

    Args:
        x (torch.Tensor): Symmetric matrix of shape `(..., N, N)`.
        eps (float, optional): Minimum allowed eigenvalue (default: `1e-5`).
        svd (bool, optional): If True, uses SVD instead of EVD (default: `True`).

    Returns:
        torch.Tensor: Rectified SPD matrix of shape `(..., N, N)`.
    """
    return SymmetricMatrixRectification.apply(x, eps, svd)


class SymmetricMatrixRectification(Function):
    r"""
    Autograd-compatible implementation of SPD matrix rectification.

    Forward:
        Clamps all eigenvalues below `eps` to `eps` and reconstructs the matrix.

    Backward:
        Uses the Loewner matrix of the clamped function:
            - Derivative is 1 where `eigval > eps`
            - Derivative is 0 where `eigval <= eps`

        Returns symmetric gradient to preserve SPD structure.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, eps: float = 1e-5, svd: bool = True
    ) -> torch.Tensor:
        if svd:
            eigvecs, eigvals, _ = torch.linalg.svd(symmetrize(x))
        else:
            eigvals, eigvecs = torch.linalg.eigh(symmetrize(x), "U")

        # max(eps, eigvals)
        f_eigvals = torch.clamp(eigvals, eps)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        ctx.eps = eps

        return symmetrize(eig2matrix(f_eigvals, eigvecs))

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors

        # Transform gradient to eigenbasis
        dx = bilinear(dy, eigvecs.mT)

        # Apply Loewner mask (0 where clamped)
        dx *= loewner(eigvals, f_eigvals, ctx.eps)

        # Transform back
        dx = bilinear(dx, eigvecs)

        # Ensure symmetric output
        return symmetrize(dx), None, None


def loewner(eigvals: torch.Tensor, f_eigvals: torch.Tensor | None = None, eps: float | None = None) -> torch.Tensor:
    r"""
    Computes the Loewner matrix for the clamped function:

        f(λ) = clamp(λ, min=eps)

    Derivative:
        df/dλ = 1.0 if λ > eps, else 0.0

    Args:
        eigvals (torch.Tensor): Original eigenvalues of shape `(..., N)`.
        f_eigvals (torch.Tensor, optional): Clamped eigenvalues. Computed if not provided.
        eps (float, optional): Minimum eigenvalue threshold used during clamping.

    Returns:
        torch.Tensor: Loewner matrix of shape `(..., N, N)`.
    """
    if f_eigvals is None:
        f_eigvals = torch.clamp(eigvals, min=eps)

    df_eigvals = (eigvals > eps).to(eigvals.dtype)

    return _loewner(eigvals, f_eigvals, df_eigvals)
