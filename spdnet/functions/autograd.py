import torch
from torch import autograd
from .inner import eig2matrix, bilinear
from .linalg import loewner_matrix

__all__ = ["SymmetricMatrixLogarithm", "SymmetricMatrixExponential", "SymmetricMatrixPower"]


class SymmetricMatrixLogarithm(autograd.Function):
    """
    Computes the logarithm of a symmetric positive definite (SPD) matrix.
    The input matrix is symmetrized before computing the logarithm.
    The output is a symmetric matrix that represents the logarithm of the input SPD matrix.
    The backward pass computes the derivative of the logarithm with respect to the input matrix.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for the logarithm of a symmetric matrix.

        Args:
            ctx: Context object to save tensors for backward pass.
            x (torch.Tensor): Input symmetric matrix of shape (..., n, n).

        Returns:
            torch.Tensor: Symmetric matrix representing the logarithm of the input matrix.
        """
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.log(eigvals)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for the logarithm of a symmetric matrix.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            dy (torch.Tensor): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input matrix.
        """
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors
        eps = torch.finfo(eigvals.dtype).eps
        df_eigvals = 1 / (eigvals + eps)
        l = loewner_matrix(eigvals, f_eigvals, df_eigvals)

        dx = torch.einsum("...ij,...jk,...kl->...il", eigvecs.mT, dy, eigvecs)
        dx = l * dx
        dx = torch.einsum("...ij,...jk,...kl->...il", eigvecs, dx, eigvecs.mT)
        return (dx + dx.mT) / 2


class SymmetricMatrixExponential(autograd.Function):
    """
    Computes the exponential of a symmetric positive definite (SPD) matrix.
    The input matrix is symmetrized before computing the exponential.
    The output is a symmetric matrix that represents the exponential of the input SPD matrix.
    The backward pass computes the derivative of the exponential with respect to the input matrix.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for the exponential of a symmetric matrix.

        Args:
            ctx: Context object to save tensors for backward pass.
            x (torch.Tensor): Input symmetric matrix of shape (..., n, n).

        Returns:
            torch.Tensor: Symmetric matrix representing the exponential of the input matrix.
        """
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.exp(eigvals)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for the exponential of a symmetric matrix.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            dy (torch.Tensor): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input matrix.
        """

        f_eigvals, eigvals, eigvecs = ctx.saved_tensors
        eps = torch.finfo(eigvals.dtype).eps
        l = loewner_matrix(eigvals, f_eigvals, f_eigvals)

        dx = bilinear(dy, eigvecs.mT)
        dx = l * dx
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2


class SymmetricMatrixPower(autograd.Function):
    """
    Computes the power of a symmetric positive definite (SPD) matrix.
    The input matrix is symmetrized before computing the power.
    The output is a symmetric matrix that represents the power of the input SPD matrix.
    The backward pass computes the derivative of the power with respect to the input matrix.
    """

    @staticmethod
    def forward(ctx, x, p):
        """
        Forward pass for the power of a symmetric matrix.

        Args:
            ctx: Context object to save tensors for backward pass.
            x (torch.Tensor): Input symmetric matrix of shape (..., n, n).
            p (float): Power to which the matrix is raised.

        Returns:
            torch.Tensor: Symmetric matrix representing the power of the input matrix.
        """
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.pow(eigvals, p)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        ctx.p = p
        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for the power of a symmetric matrix.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            dy (torch.Tensor): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            tuple: Gradient of the loss with respect to the input matrix and None for p.
        """
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors
        eps = torch.finfo(eigvals.dtype).eps

        df_eigvals = ctx.p * f_eigvals / (eigvals + eps)
        l = loewner_matrix(eigvals, f_eigvals, df_eigvals)

        dx = bilinear(dy, eigvecs.mT)
        dx = l * dx
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2, None


class SymmetricMatrixRectification(autograd.Function):
    """
    Computes the rectification of a symmetric matrix to ensure it is positive definite.
    The input matrix is symmetrized before applying the rectification.
    The output is a symmetric matrix that represents the rectified input matrix.
    The backward pass computes the derivative of the rectification with respect to the input matrix.
    """

    @staticmethod
    def forward(ctx, x, eps=1e-5):
        """
        Forward pass for the rectification of a symmetric matrix.
        Args:
            ctx: Context object to save tensors for backward pass.
            x (torch.Tensor): Input symmetric matrix of shape (..., n, n).
            eps (float): Small value to ensure numerical stability.

        Returns:
            torch.Tensor: Symmetric matrix representing the rectified input matrix.
        """
        x = (x + x.mT) / 2
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.clamp(eigvals, eps)
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        ctx.eps = eps
        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for the rectification of a symmetric matrix.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            dy (torch.Tensor): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            tuple: Gradient of the loss with respect to the input matrix and None for eps.
        """
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors

        df_eigvals = torch.where(eigvals > ctx.eps, 1, 0)
        l = loewner_matrix(eigvals, f_eigvals, df_eigvals)

        dx = bilinear(dy, eigvecs.mT)
        dx = l * dx
        dx = bilinear(dx, eigvecs)
        return (dx + dx.mT) / 2, None
