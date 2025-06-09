import torch
from torch.autograd import Function

__all__ = ["bilinear"]


def bilinear(x, z):
    """
    Computes broadcasted bilinear mapping `z @ x @ z^T`.

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., n, n).
        z (torch.Tensor): Mapping matrix of shape (..., m, n).

    Returns:
        torch.Tensor: Transformed SPD matrix of shape (..., m, m).
    """
    return Bilinear.apply(x, z)


class Bilinear(Function):
    """
    Custom autograd function for bilinear mapping of SPD matrices.
    Computes z @ x @ z^T where x is an SPD matrix and z is a mapping matrix.
    """

    @staticmethod
    def forward(ctx, x, z):
        ctx.save_for_backward(x, z)
        return z @ x @ z.mT

    @staticmethod
    def backward(ctx, dy):
        x, z = ctx.saved_tensors
        dx = z.mT @ dy @ z
        dz = (dy + dy.mT) @ z @ x
        return dx, dz
