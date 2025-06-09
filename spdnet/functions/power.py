import torch
from torch.autograd import Function

from .bilinear import bilinear
from .linalg import eig2matrix
from .utils import loewner as _loewner

__all__ = ["sym_mat_pow", "sym_mat_square", "sym_mat_sqrt", "sym_mat_inv"]


def sym_mat_pow(x: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Computes the matrix power `x^p` for a symmetric positive definite (SPD) matrix `x`.

    This uses the eigendecomposition of `x`:
        x^p = Q @ diag(λ^p) @ Q.T
    where x = Q @ diag(λ) @ Q.T and λ are the eigenvalues of `x`.

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.
        p (float): Exponent to which the matrix is raised.

    Returns:
        torch.Tensor: Matrix power `x^p` of shape `(..., N, N)`.
    """
    return SymmetricMatrixPower.apply(x, p)


def sym_mat_square(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the matrix square `x^2` for a symmetric matrix.

    Equivalent to `sym_mat_pow(x, 2)`.

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.

    Returns:
        torch.Tensor: Matrix square of shape `(..., N, N)`.
    """
    return sym_mat_pow(x, 2)


def sym_mat_sqrt(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the matrix square root `x^{1/2}` for a symmetric matrix.

    Equivalent to `sym_mat_pow(x, 0.5)`.

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.

    Returns:
        torch.Tensor: Matrix square root of shape `(..., N, N)`.
    """
    return sym_mat_pow(x, 0.5)


def sym_mat_inv(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the matrix inverse `x^{-1}` for a symmetric matrix.

    Equivalent to `sym_mat_pow(x, -1)`.

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.

    Returns:
        torch.Tensor: Matrix inverse of shape `(..., N, N)`.
    """
    return sym_mat_pow(x, -1)


class SymmetricMatrixPower(Function):
    r"""
    Autograd-compatible implementation of matrix power `x^p` for symmetric matrices.

    Forward:
        Computes `x^p = Q @ diag(λ^p) @ Q.T` using eigendecomposition.

    Backward:
        Uses the Loewner matrix to compute:
            ∂x^p/∂x = V @ [L ⊙ (Vᵀ ∂Y V)] @ Vᵀ
        and optionally:
            ∂x^p/∂p = sum_i (∂λ_i^p/∂p * ∂Y/∂λ_i)

    Notes:
        This implementation assumes `x` is symmetric and positive definite.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, p: float) -> torch.Tensor:
        x = (x + x.mT) / 2  # Ensure symmetry
        eigvals, eigvecs = torch.linalg.eigh(x)
        f_eigvals = torch.pow(eigvals, p)

        ctx.save_for_backward(f_eigvals, eigvals, eigvecs)
        ctx.p = p

        return eig2matrix(f_eigvals, eigvecs)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, dy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        f_eigvals, eigvals, eigvecs = ctx.saved_tensors
        p = ctx.p

        # Compute dX using Loewner matrix
        dx = bilinear(dy, eigvecs.mT)
        dx *= loewner(eigvals, f_eigvals, p)
        dx = bilinear(dx, eigvecs)
        dx = (dx + dx.mT) / 2  # Ensure gradient is symmetric

        # Optional gradient w.r.t. power `p` (if needed)
        dp = None
        if torch.jit.isinstance(p, torch.Tensor) and p.requires_grad:
            log_eigvals = torch.log(eigvals)
            dp_coeffs = f_eigvals * log_eigvals
            d_eig = bilinear(dy, eigvecs.mT)
            diag_d_eig = torch.diagonal(d_eig, dim1=-2, dim2=-1)
            dp = torch.sum(diag_d_eig * dp_coeffs, dim=-1)

        return dx, dp


def loewner(eigvals: torch.Tensor, f_eigvals: torch.Tensor | None = None, p: float | None = None) -> torch.Tensor:
    r"""
    Computes the Loewner matrix for the matrix power function `λ^p`.

    For `i ≠ j`:     L_ij = (λ_i^p - λ_j^p) / (λ_i - λ_j)
    For `i == j`:    L_ii = p * λ_i^{p - 1}

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape `(..., N)`.
        f_eigvals (torch.Tensor, optional): Precomputed λ^p values. If None, computed.
        p (float, optional): Power `p` for the derivative rule.

    Returns:
        torch.Tensor: Loewner matrix of shape `(..., N, N)`.
    """
    if f_eigvals is None:
        f_eigvals = torch.pow(eigvals, p)

    eps = torch.finfo(eigvals.dtype).eps
    df_eigvals = p * f_eigvals / (eigvals + eps)

    return _loewner(eigvals, f_eigvals, df_eigvals)
