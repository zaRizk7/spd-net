import torch
from torch.autograd import Function

from .bilinear import bilinear
from .linalg import eig2matrix
from .utils import loewner as _loewner

__all__ = ["sym_mat_log"]


def sym_mat_log(x):
    """
    Compute the matrix logarithm of a symmetric positive definite matrix
    using its eigendecomposition.

    For a symmetric matrix `X`, this computes:
        log(X) = Q @ diag(log(λ)) @ Q.T
    where X = Q @ diag(λ) @ Q.T is the eigendecomposition of X,
    and log(λ) is applied elementwise.

    Args:
        x (torch.Tensor): Symmetric positive definite matrix of shape (..., N, N).

    Returns:
        torch.Tensor: Matrix logarithm of `x`, of shape (..., N, N).
    """
    return SymmetricMatrixLogarithm.apply(x)


class SymmetricMatrixLogarithm(Function):
    """
    Autograd-compatible implementation of the matrix logarithm for symmetric
    positive definite matrices using eigendecomposition.

    The backward pass uses the Loewner matrix to compute the gradient.
    """

    @staticmethod
    def forward(ctx, x):
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.log(eigvals)

        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(ctx, dy):
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors
        dx = bilinear(dy, eigvecs.mT)
        dx *= loewner(eigvals, f_eigvals)
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2


def loewner(eigvals, f_eigvals=None):
    """
    Compute the Loewner matrix L_{ij} = (f(λ_i) - f(λ_j)) / (λ_i - λ_j)
    and L_{ii} = 1 / λ_i for the logarithm function f(λ) = log(λ).

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape (..., N).
        f_eigvals (torch.Tensor, optional): Precomputed log eigenvalues. If None, computed as log(eigvals).

    Returns:
        torch.Tensor: Loewner matrix of shape (..., N, N).
    """
    if f_eigvals is None:
        f_eigvals = torch.log(eigvals)

    eps = torch.finfo(eigvals.dtype).eps
    df_eigvals = 1 / (eigvals + eps)

    return _loewner(eigvals, f_eigvals, df_eigvals)
