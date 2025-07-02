import geoopt
import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from ..functions import bilinear


__all__ = ["BiMap"]


class BiMap(nn.Module):
    r"""
    Bilinear mapping layer for symmetric positive definite (SPD) matrices.

    This layer performs a transformation of the form:

        Z = W @ X @ Wᵀ

    where `W` is an orthogonally-constrained learnable weight matrix, and
    `X` is an SPD input matrix. The operation is analogous to `nn.Linear`
    but maps one SPD matrix to another via a bilinear form that preserves
    the SPD structure when `W` is orthogonal.

    Args:
        in_spatial (int):
            Input SPD matrix size. Input should be of shape `(*, in_spatial, in_spatial)`.
        out_spatial (int):
            Output SPD matrix size. Output will be of shape `(*, out_spatial, out_spatial)`.
        device (torch.device, optional):
            The device on which to create the weight matrix.
        dtype (torch.dtype, optional):
            Data type for the weight matrix.

    Shape:
        - Input: `(*, in_spatial, in_spatial)`
        - Output: `(*, out_spatial, out_spatial)`

    Attributes:
        weight (torch.Tensor):
            Orthogonally-constrained learnable weight matrix of shape
            `(out_spatial, in_spatial)` or its transpose, depending on projection type.
    """

    __constants__ = ["in_spatial", "out_spatial"]
    in_spatial: int
    out_spatial: int
    weight: torch.Tensor

    def __init__(self, in_spatial, out_spatial, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_spatial = in_spatial
        self.out_spatial = out_spatial
        self.is_upscale = out_spatial > in_spatial

        self.weight_shape = (out_spatial, in_spatial) if self.is_upscale else (in_spatial, out_spatial)

        manifold = geoopt.Stiefel()
        p, n = self.weight_shape
        weight = manifold.random(p, n, **factory_kwargs)
        self._weight = geoopt.ManifoldParameter(weight, manifold)

    @property
    def weight(self):
        """Returns the appropriate weight matrix (transposed if downscaling)."""
        return self._weight if self.is_upscale else self._weight.mT

    def __repr__(self):
        return f"{self.__class__.__name__}(in_spatial={self.in_spatial}, out_spatial={self.out_spatial})"

    def reset_parameters(self):
        """
        Re-initialize the weight matrix using uniform sampling from the Stiefel manifold.
        """
        factory_kwargs = {"device": self._weight.device, "dtype": self._weight.dtype}
        sample = self._weight.manifold.random
        p, n = self.weight_shape
        self._weight.data = sample(p, n, **factory_kwargs)

    def forward(self, x):
        """
        Apply the bilinear transformation: `Z = W @ X @ Wᵀ`.

        Args:
            x (torch.Tensor):
                Input SPD matrix of shape `(*, in_spatial, in_spatial)`.

        Returns:
            torch.Tensor:
                Output SPD matrix of shape `(*, out_spatial, out_spatial)`.
        """
        return bilinear(x, self.weight)
