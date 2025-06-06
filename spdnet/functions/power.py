import opt_einsum as oe
import torch
from torch.autograd import Function

from .inner import bilinear, eig2matrix
from .utils import loewner as _loewner

__all__ = ["sym_mat_pow", "sym_mat_square", "sym_mat_sqrt", "sym_mat_inv"]


def sym_mat_pow(x, p):
    """
    Compute the matrix power of a symmetric matrix using eigendecomposition.

    For a symmetric matrix `X`, this computes:
        X^p = Q @ diag(λ^p) @ Q.T
    where X = Q @ diag(λ) @ Q.T is the eigendecomposition of X,
    and λ^p is applied elementwise.

    Args:
        x (torch.Tensor): Symmetric positive definite matrix of shape (..., N, N).
        p (float): Power to which the matrix should be raised.

    Returns:
        torch.Tensor: Matrix raised to the power `p`, of shape (..., N, N).
    """
    return SymmetricMatrixPower.apply(x, p)


def sym_mat_square(x):
    """
    Compute the square of a symmetric matrix using eigendecomposition.

    Equivalent to sym_mat_pow(x, 2).

    Args:
        x (torch.Tensor): Symmetric positive definite matrix of shape (..., N, N).

    Returns:
        torch.Tensor: Matrix squared, of shape (..., N, N).
    """
    return sym_mat_pow(x, 2)


def sym_mat_sqrt(x):
    """
    Compute the matrix square root of a symmetric matrix using eigendecomposition.

    Equivalent to sym_mat_pow(x, 0.5).

    Args:
        x (torch.Tensor): Symmetric positive definite matrix of shape (..., N, N).

    Returns:
        torch.Tensor: Matrix square root, of shape (..., N, N).
    """
    return sym_mat_pow(x, 0.5)


def sym_mat_inv(x):
    """
    Compute the matrix inverse of a symmetric matrix using eigendecomposition.

    Equivalent to sym_mat_pow(x, -1).

    Args:
        x (torch.Tensor): Symmetric positive definite matrix of shape (..., N, N).

    Returns:
        torch.Tensor: Matrix inverse, of shape (..., N, N).
    """
    return sym_mat_pow(x, -1)


class SymmetricMatrixPower(Function):
    """
    Autograd-compatible implementation of matrix power for symmetric matrices
    using eigendecomposition and the Loewner matrix for gradient computation.
    """

    @staticmethod
    def forward(ctx, x, p):
        return forward(x, p, ctx)

    @staticmethod
    def backward(ctx, dy):
        return backward(dy, *ctx.saved_tensors, ctx.p)


def forward(x, p, ctx=None):
    """
    Forward pass for computing matrix power using eigendecomposition.

    Args:
        x (torch.Tensor): Symmetric positive definite matrix of shape (..., N, N).
        p (float): Power to which the matrix should be raised.
        ctx (torch.autograd.function.FunctionCtx, optional): Autograd context.

    Returns:
        torch.Tensor: Matrix raised to the power `p`, of shape (..., N, N).
    """
    x = (x + x.mT) / 2
    eigvals, eigvecs = torch.linalg.eigh(x)
    f_eigvals = torch.pow(eigvals, p)
    if ctx is not None:
        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        ctx.p = p
    return eig2matrix(f_eigvals, eigvecs)


def backward(dy, f_eigvals, eigvals, eigvecs, p):
    """
    Backward pass for matrix power with conditional gradient w.r.t. power p.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: Gradient w.r.t. input matrix and (optional) power p.
    """
    dx = bilinear(dy, eigvecs.mT)
    dx *= loewner(eigvals, f_eigvals, p)
    dx = bilinear(dx, eigvecs)
    dx = (dx + dx.mT) / 2

    dp = None
    if isinstance(p, torch.Tensor) and p.requires_grad:
        log_eigvals = torch.log(eigvals)
        dp_coeffs = f_eigvals * log_eigvals
        d_eig = bilinear(dy, eigvecs.mT)
        diag_d_eig = torch.diagonal(d_eig, dim1=-2, dim2=-1)
        dp = oe.contract("...i,...i->...", diag_d_eig, dp_coeffs)

    return dx, dp


def loewner(eigvals, f_eigvals=None, p=None):
    """
    Compute the Loewner matrix L_{ij} = (λ_i^p - λ_j^p) / (λ_i - λ_j)
    and L_{ii} = p * λ_i^(p-1) for differentiating the matrix power.

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape (..., N).
        f_eigvals (torch.Tensor, optional): λ^p, the function applied to the eigenvalues.
        p (float, optional): Power to which eigenvalues were raised.

    Returns:
        torch.Tensor: Loewner matrix of shape (..., N, N).
    """
    if f_eigvals is None:
        f_eigvals = torch.pow(eigvals, p)

    eps = torch.finfo(eigvals.dtype).eps
    df_eigvals = p * f_eigvals / (eigvals + eps)

    return _loewner(eigvals, f_eigvals, df_eigvals)
