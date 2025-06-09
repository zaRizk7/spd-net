import torch
from torch.autograd import Function

from .bilinear import bilinear
from .linalg import eig2matrix
from .utils import loewner as _loewner

__all__ = ["sym_mat_rec"]


def sym_mat_rec(x, eps=1e-5):
    """
    Perform rectification on a symmetric matrix by clamping its eigenvalues to a minimum threshold.

    Given a symmetric matrix `X`, this computes:
        X_rect = Q @ diag(clamp(λ, min=eps)) @ Q.T
    where X = Q @ diag(λ) @ Q.T is the eigendecomposition of X.

    This ensures positive definiteness and is commonly used to prevent degenerate
    or near-singular matrices in SPD-based models.

    Args:
        x (torch.Tensor): Symmetric matrix of shape (..., N, N).
        eps (float, optional): Minimum eigenvalue threshold. Default is 1e-5.

    Returns:
        torch.Tensor: Rectified matrix with the same shape as input (..., N, N).
    """
    return SymmetricMatrixRectification.apply(x, eps)


class SymmetricMatrixRectification(Function):
    """
    Autograd-compatible implementation of symmetric matrix rectification via eigenvalue clamping.

    This operation clamps all eigenvalues below `eps` to `eps` and reconstructs the matrix.
    The backward pass computes the gradient using the Loewner matrix.
    """

    @staticmethod
    def forward(ctx, x, eps=1e-5):
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.clamp(eigvals, eps)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        ctx.eps = eps
        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(ctx, dy):
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors
        dx = bilinear(dy, eigvecs.mT)
        dx *= loewner(eigvals, f_eigvals, ctx.eps)
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2, None


def loewner(eigvals, f_eigvals=None, eps=None):
    """
    Compute the Loewner matrix L_{ij} = (f(λ_i) - f(λ_j)) / (λ_i - λ_j)
    for the rectification function f(λ) = clamp(λ, min=eps).

    The derivative df is 1 when λ > eps and 0 otherwise.

    Args:
        eigvals (torch.Tensor): Original eigenvalues of shape (..., N).
        f_eigvals (torch.Tensor, optional): Clamped eigenvalues of shape (..., N).
        eps (float, optional): Threshold for clamping.

    Returns:
        torch.Tensor: Loewner matrix of shape (..., N, N).
    """
    if f_eigvals is None:
        f_eigvals = torch.clamp(eigvals, eps)

    df_eigvals = torch.where(eigvals > eps, 1.0, 0.0)

    return _loewner(eigvals, f_eigvals, df_eigvals)
