import torch

__all__ = ["eig2matrix", "trace", "fro"]


def eig2matrix(eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
    r"""
    Reconstructs a symmetric positive definite (SPD) matrix from its eigendecomposition.

    Given eigenvalues `eigvals` and eigenvectors `eigvecs`, computes:

        A = V @ diag(eigvals) @ V.T

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape `(..., n)`.
        eigvecs (torch.Tensor): Eigenvectors of shape `(..., n, n)`.

    Returns:
        torch.Tensor: Reconstructed SPD matrix of shape `(..., n, n)`.
    """
    # Expand eigvals to diagonal matrix and perform V @ diag(λ) @ V.T
    return torch.matmul(eigvecs * eigvals[..., None, :], eigvecs.mT)


def trace(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the trace of a batched SPD matrix.

    Args:
        x (torch.Tensor): Input tensor of shape `(..., n, n)`.

    Returns:
        torch.Tensor: Trace of each matrix, shape `(...)`.
    """
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(dim=-1)


def fro(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the Frobenius norm of a batched SPD matrix.

    The Frobenius norm is defined as:

        ||X||_F = sqrt(sum_{i,j} X_{ij}^2)

    Args:
        x (torch.Tensor): Input tensor of shape `(..., n, n)`.

    Returns:
        torch.Tensor: Frobenius norm of each matrix, shape `(...)`.
    """
    return torch.linalg.norm(x, ord="fro", dim=(-2, -1))
