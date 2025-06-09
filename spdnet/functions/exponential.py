import torch
from torch.autograd import Function

from .bilinear import bilinear
from .linalg import eig2matrix
from .utils import loewner as _loewner

__all__ = ["sym_mat_exp"]


def sym_mat_exp(x):
    """
    Compute the matrix exponential of a symmetric matrix using its eigendecomposition.

    Given a symmetric matrix `X`, this computes:
        exp(X) = Q @ diag(exp(λ)) @ Q.T
    where X = Q @ diag(λ) @ Q.T is the eigendecomposition of X.

    Args:
        x (torch.Tensor): Symmetric matrix of shape (..., N, N).

    Returns:
        torch.Tensor: Matrix exponential of `x`, of shape (..., N, N).
    """
    return SymmetricMatrixExponential.apply(x)


class SymmetricMatrixExponential(Function):
    """
    Autograd-compatible implementation of matrix exponential for symmetric matrices
    using eigendecomposition and the Loewner matrix for gradient computation.
    """

    @staticmethod
    def forward(ctx, x):
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.exp(eigvals)
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
    for the exponential function, where f(λ) = exp(λ).

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape (..., N).
        f_eigvals (torch.Tensor, optional): Function values at eigenvalues. If None, computed as exp(eigvals).

    Returns:
        torch.Tensor: Loewner matrix of shape (..., N, N).
    """
    if f_eigvals is None:
        f_eigvals = torch.exp(eigvals)
    return _loewner(eigvals, f_eigvals, f_eigvals)
