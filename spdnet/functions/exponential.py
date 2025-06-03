import torch
from torch.autograd import Function

from .inner import bilinear, eig2matrix
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
        return forward(x, ctx)

    @staticmethod
    def backward(ctx, dy):
        return backward(dy, *ctx.saved_tensors)


def forward(x, ctx=None):
    """
    Forward pass for the matrix exponential of a symmetric matrix.

    Args:
        x (torch.Tensor): Symmetric matrix of shape (..., N, N).
        ctx (torch.autograd.function.FunctionCtx, optional): Autograd context to save tensors.

    Returns:
        torch.Tensor: Matrix exponential of `x`, of shape (..., N, N).
    """
    x = (x + x.mT) / 2  # Ensure symmetry
    eigvals, eigvecs = torch.linalg.eigh(x)
    f_eigvals = torch.exp(eigvals)
    if ctx is not None:
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
    return eig2matrix(f_eigvals, eigvecs)


def backward(dy, f_eigvals, eigvals, eigvecs):
    """
    Backward pass for the matrix exponential using the Loewner matrix.

    Args:
        dy (torch.Tensor): Upstream gradient of shape (..., N, N).
        f_eigvals (torch.Tensor): Exponentiated eigenvalues, shape (..., N).
        eigvals (torch.Tensor): Original eigenvalues, shape (..., N).
        eigvecs (torch.Tensor): Eigenvectors, shape (..., N, N).

    Returns:
        torch.Tensor: Gradient with respect to the input matrix, shape (..., N, N).
    """
    dx = bilinear(dy, eigvecs.mT)
    dx *= loewner(eigvals, f_eigvals)
    dx = bilinear(dx, eigvecs)
    return (dx + dx.mT) / 2  # Ensure symmetry


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
