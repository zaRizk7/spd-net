import torch
from torch import nn

from .activations import matrix_exponential, matrix_logarithm

__all__ = ["aritmetic_mean"]


def aritmetic_mean(*x):
    """
    Compute the arithmetic mean of a batch of SPD matrices.

    Args:
        x (torch.Tensor): Input tensor of shape (..., n, n) where n is the spatial dimension.
        dim (int): Dimension along which to compute the mean.

    Returns:
        torch.Tensor: Tensor with the arithmetic mean of the input matrices.
    """
    x = torch.stack(x, 0)
    x = matrix_logarithm(x)
    x = torch.mean(x, 0)
    x = matrix_exponential(x)
    return x
