import torch

__all__ = ["eig2matrix", "trace", "fro"]


def eig2matrix(eigvals, eigvecs):
    """
    Convert eigvals and eigvecs to a SPD matrix.

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape (..., n).
        eigvecs (torch.Tensor): Eigenvectors of shape (..., n, n).

    Returns:
        torch.Tensor: SPD matrix of shape (..., n, n).
    """
    return eigvecs * eigvals[..., None, :] @ eigvecs.mT


def trace(x):
    """
    Computes the trace of a batched SPD matrix.

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., n, n).

    Returns:
        torch.Tensor: Trace of the matrix, shape (...,).
    """
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(dim=-1)


def fro(x):
    """
    Computes the Frobenius norm of a batched SPD matrix.

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., n, n).

    Returns:
        torch.Tensor: Frobenius norm of the matrix, shape (...,).
    """
    return torch.sqrt(torch.sum(x * x, dim=(-2, -1)))
