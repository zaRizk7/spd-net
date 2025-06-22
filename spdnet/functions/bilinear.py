import torch
from torch.autograd import Function

__all__ = ["bilinear"]


def bilinear(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    r"""
    Applies a broadcasted bilinear transformation to an SPD matrix.

    Given an input SPD matrix `x` of shape `(..., n, n)` and a mapping matrix `z`
    of shape `(..., m, n)`, this function computes the bilinear form:

        `z @ x @ z.T`

    where `@` denotes matrix multiplication. This operation preserves symmetry
    and is used in many SPD-based geometric deep learning settings.

    Args:
        x (torch.Tensor): Symmetric positive definite matrix of shape `(..., n, n)`.
        z (torch.Tensor): Mapping matrix of shape `(..., m, n)`.

    Returns:
        torch.Tensor: Output matrix of shape `(..., m, m)`, symmetric and positive semi-definite.
    """
    return Bilinear.apply(x, z)


class Bilinear(Function):
    r"""
    Custom autograd function for the bilinear form `z @ x @ z.T`.

    This function supports autograd and can be JIT-compiled.

    Forward:
        Computes `z @ x @ z.T` where `x` is an SPD matrix and `z` is a projection matrix.

    Backward:
        Given gradient `dy` of shape `(..., m, m)`, computes:
            - `dx = z.T @ dy @ z`
            - `dz = (dy + dy.T) @ z @ x`
        Note: Symmetrization of `dy` ensures gradient consistency due to symmetry of `x`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, z)
        return torch.matmul(torch.matmul(z, x), z.mT)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, z = ctx.saved_tensors

        # Allocate for more efficient memory usage
        dx = torch.empty_like(x)
        dz = torch.empty_like(z)

        # dx = z.mT @ dy @ z
        torch.matmul(torch.matmul(z.mT, dy), z, out=dx)
        # dz = (dy + dy.mT) @ z @ x
        torch.matmul(torch.matmul(dy + dy.mT, z), x, out=dz)

        return dx, dz
