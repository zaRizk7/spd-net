import torch
from torch.autograd import Function

from .inner import bilinear, eig2matrix
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
        return forward(x, ctx)

    @staticmethod
    def backward(ctx, dy):
        return backward(dy, *ctx.saved_tensors)


def forward(x, ctx=None):
    """
    Forward pass for the matrix logarithm using eigendecomposition.

    Args:
        x (torch.Tensor): Symmetric positive definite matrix of shape (..., N, N).
        ctx (torch.autograd.function.FunctionCtx, optional): Autograd context.

    Returns:
        torch.Tensor: Matrix logarithm of `x`, of shape (..., N, N).
    """
    x = (x + x.mT) / 2  # Ensure symmetry
    eigvals, eigvecs = torch.linalg.eigh(x)
    f_eigvals = torch.log(eigvals)
    if ctx is not None:
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
    return eig2matrix(f_eigvals, eigvecs)


def backward(dy, f_eigvals, eigvals, eigvecs):
    """
    Backward pass for the matrix logarithm using the Loewner matrix.

    Args:
        dy (torch.Tensor): Upstream gradient of shape (..., N, N).
        f_eigvals (torch.Tensor): Logarithm of eigenvalues, shape (..., N).
        eigvals (torch.Tensor): Eigenvalues from forward pass, shape (..., N).
        eigvecs (torch.Tensor): Eigenvectors from forward pass, shape (..., N, N).

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
    df_eigvals = 1 / (eigvals + eps)  # Elementwise derivative of log(λ)

    return _loewner(eigvals, f_eigvals, df_eigvals)
