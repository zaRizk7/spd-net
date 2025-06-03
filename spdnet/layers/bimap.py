import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.parametrizations import orthogonal

from ..functions import bilinear

__all__ = ["BiMap"]


class BiMap(nn.Module):
    """
    Downscale/upscale spatial dimensions of SPD matrices by using bilinear
    mapping. The BiMap layer is similar to nn.Linear, but enforces orthogonality
    on the weight matrix, ensuring the weight space is within a non-compact
    Stiefel manifold. Transforms both rows and columns of the input SPD matrix.

    Args:
        in_spatial (int): Input spatial dimension.
        out_spatial (int): Output spatial dimension.

    Shape:
        - input: (*, in_spatial, in_spatial)
        - output: (*, out_spatial, out_spatial)

    Attributes:
        - weight (torch.Tensor): The learnable weight of shape (in_spatial, out_spatial).
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

        weight = torch.empty(out_spatial, in_spatial, **factory_kwargs)
        self.weight = nn.Parameter(weight)
        self.reset_parameters()
        orthogonal(self)

    def reset_parameters(self):
        init.orthogonal_(self.weight)

    def forward(self, x):
        return bilinear(x, self.weight)
