import torch
from torch import autograd

from .inner import bilinear, eig2matrix
from .linalg import loewner_matrix

__all__ = [
    "SymmetricMatrixLogarithm",
    "SymmetricMatrixExponential",
    "SymmetricMatrixPower",
    "SymmetricMatrixRectification",
]


class SymmetricMatrixLogarithm(autograd.Function):
    r"""
    Computes the matrix logarithm of a symmetric input matrix using its eigendecomposition.

    This operation is defined as:
        log(X) = Q @ diag(log(λ)) @ Q.T
    where X = Q @ diag(λ) @ Q.T is the eigendecomposition of the symmetric matrix X.

    The backward pass computes the gradient via the Loewner matrix.

    Args:
        x (torch.Tensor): A symmetric matrix of shape (..., N, N).

    Returns:
        torch.Tensor: The matrix logarithm of `x` with the same shape (..., N, N).
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
        eps = torch.finfo(eigvals.dtype).eps
        df_eigvals = 1 / (eigvals + eps)
        L = loewner_matrix(eigvals, f_eigvals, df_eigvals)

        dx = bilinear(dy, eigvecs.mT)
        dx *= L
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2


class SymmetricMatrixExponential(autograd.Function):
    r"""
    Computes the matrix exponential of a symmetric input matrix using its eigendecomposition.

    This operation is defined as:
        exp(X) = Q @ diag(exp(λ)) @ Q.T
    where X = Q @ diag(λ) @ Q.T is the eigendecomposition of the symmetric matrix X.

    Args:
        x (torch.Tensor): A symmetric matrix of shape (..., N, N).

    Returns:
        torch.Tensor: The matrix exponential of `x` with the same shape (..., N, N).
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
        L = loewner_matrix(eigvals, f_eigvals, f_eigvals)

        dx = bilinear(dy, eigvecs.mT)
        dx *= L
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2


class SymmetricMatrixPower(autograd.Function):
    r"""
    Computes the power of a symmetric matrix via eigendecomposition.

    This operation is defined as:
        X^p = Q @ diag(λ^p) @ Q.T
    where X = Q @ diag(λ) @ Q.T is the eigendecomposition of the symmetric matrix X.

    Args:
        x (torch.Tensor): A symmetric matrix of shape (..., N, N).
        p (float): The power to raise the matrix to.

    Returns:
        torch.Tensor: The result of raising `x` to the power `p`, with shape (..., N, N).
    """

    @staticmethod
    def forward(ctx, x, p):
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.pow(eigvals, p)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        ctx.p = p
        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(ctx, dy):
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors
        eps = torch.finfo(eigvals.dtype).eps

        df_eigvals = ctx.p * f_eigvals / (eigvals + eps)
        L = loewner_matrix(eigvals, f_eigvals, df_eigvals)

        dx = bilinear(dy, eigvecs.mT)
        dx *= L
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2, None


class SymmetricMatrixRectification(autograd.Function):
    r"""
    Performs eigenvalue rectification on a symmetric matrix to ensure numerical stability.

    This operation clamps eigenvalues to a minimum value `eps`:
        rect(X) = Q @ diag(max(λ, eps)) @ Q.T

    Useful for enforcing positive-definiteness of nearly semi-definite matrices.

    Args:
        x (torch.Tensor): A symmetric matrix of shape (..., N, N).
        eps (float, optional): Minimum eigenvalue threshold. Default is 1e-5.

    Returns:
        torch.Tensor: Rectified matrix with eigenvalues clamped from below.
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

        df_eigvals = torch.where(eigvals > ctx.eps, 1, 0)
        L = loewner_matrix(eigvals, f_eigvals, df_eigvals)

        dx = bilinear(dy, eigvecs.mT)
        dx *= L
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2, None
