import torch
from torch import linalg as la
from torch import nn

__all__ = ["EigenActivation"]

EINSUM_PATTERN = "...ij,...j,...kj->...ik"


def matrix_rectification(x, eps=1e-5):
    """
    Rectify the eigenvalues of a SPD matrix to ensure they are non-negative.
    Used to apply non-linearity while also preventing degeneracy in the SPD space.

    Args:
        x (torch.Tensor): Input tensor of shape (..., n, n) where n is the spatial dimension.
        eps (float): Small value to ensure numerical stability.

    Returns:
        torch.Tensor: Tensor with rectified eigenvalues.
    """
    eigenvalues, eigenvectors = la.eigh(x)
    eigenvalues = torch.maximum(eigenvalues, eps)
    return torch.einsum(EINSUM_PATTERN, eigenvectors, eigenvalues, eigenvectors)


def matrix_exponential(x):
    """
    Compute the exponential of the eigenvalues of a SPD matrix.

    Args:
        x (torch.Tensor): Input tensor of shape (..., n, n) where n is the spatial dimension.

    Returns:
        torch.Tensor: Tensor with exponential of eigenvalues.
    """
    eigenvalues, eigenvectors = la.eigh(x)
    return torch.einsum(
        EINSUM_PATTERN, eigenvectors, torch.exp(eigenvalues), eigenvectors
    )


def matrix_logarithm(x):
    """
    Compute the logarithm of the eigenvalues of a SPD matrix.

    Args:
        x (torch.Tensor): Input tensor of shape (..., n, n) where n is the spatial dimension.

    Returns:
        torch.Tensor: Tensor with logarithm of eigenvalues.
    """
    eigenvalues, eigenvectors = la.eigh(x)
    return torch.einsum(
        EINSUM_PATTERN, eigenvectors, torch.log(eigenvalues), eigenvectors
    )


class EigenActivation(nn.Module):
    """
    Applies an activation function to the eigenvalues of a SPD matrix.

    Args:
        activation (str): The activation function to apply to the eigenvalues. Options are 'rectify', 'log', or 'exp'.
        eps (float): Small value to ensure numerical stability for 'rectify' activation function.
    """

    __constants__ = ["activation", "eps"]
    activation: str
    eps: float

    def __init__(self, activation="rectify", eps=1e-5):
        if activation not in ("rectify", "log", "exp"):
            msg = f"activation must be one of 'rectify', 'log', or 'exp'. Got '{activation}'."
            raise ValueError(msg)

        super().__init__()
        self.activation = activation
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(self, x):
        if self.activation == "rectify":
            return matrix_rectification(x, self.eps)
        elif self.activation == "log":
            return matrix_logarithm(x)
        return matrix_exponential(x)
