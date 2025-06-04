import opt_einsum as oe
import torch

__all__ = ["eig2matrix", "bdot", "bilinear", "trace", "fro"]


def eig2matrix(eigvals, eigvecs):
    """
    Convert eigvals and eigvecs to a SPD matrix.

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape (..., n).
        eigvecs (torch.Tensor): Eigenvectors of shape (..., n, n).

    Returns:
        torch.Tensor: SPD matrix of shape (..., n, n).
    """
    return oe.contract("...ij,...j,...kj->...ik", eigvecs, eigvals, eigvecs)


def bdot(x, z):
    """
    Computes the dot product of broadcasted matrices `x @ z`.

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., n, n).
        z (torch.Tensor): Vector or matrix of shape (..., n, m).

    Returns:
        torch.Tensor: Batched dot product of shape (..., n, m).
    """
    return oe.contract("...ij,...jk->...ik", x, z)


def bilinear(x, z):
    """
    Computes broadcasted bilinear mapping `z @ x @ z^T`.

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., n, n).
        z (torch.Tensor): Mapping matrix of shape (..., m, n).

    Returns:
        torch.Tensor: Transformed SPD matrix of shape (..., m, m).
    """
    return oe.contract("...ki,...ij,...lj->...kl", z, x, z)


def trace(x):
    """
    Computes the trace of a batched SPD matrix.

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., n, n).

    Returns:
        torch.Tensor: Trace of the matrix, shape (...,).
    """
    return oe.contract("...ii->...", x)


def fro(x):
    """
    Computes the Frobenius norm of a batched SPD matrix.

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., n, n).

    Returns:
        torch.Tensor: Frobenius norm of the matrix, shape (...,).
    """
    return torch.sqrt(oe.contract("...ij,...ij->...", x, x))
