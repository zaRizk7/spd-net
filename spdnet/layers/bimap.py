import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.parametrizations import orthogonal

from ..functions import bilinear

__all__ = ["BiMap"]


class BiMap(nn.Module):
    r"""
    Bilinear mapping layer for symmetric positive definite (SPD) matrices.

    This layer performs a transformation of the form:
        Z = W @ X @ W^T,
    where `W` is an orthogonally parametrized learnable weight matrix,
    and `X` is an SPD input matrix.

    The operation is analogous to `nn.Linear` but respects the SPD manifold geometry
    by projecting the SPD input into a lower or higher-dimensional SPD space via
    orthogonal bilinear transformation.

    Args:
        in_spatial (int): Input SPD matrix size (X of shape `(*, in_spatial, in_spatial)`).
        out_spatial (int): Output SPD matrix size (Z of shape `(*, out_spatial, out_spatial)`).
        device (torch.device, optional): Device to place the weight parameter.
        dtype (torch.dtype, optional): Data type for the weight parameter.

    Shape:
        - Input: (*, in_spatial, in_spatial)
        - Output: (*, out_spatial, out_spatial)

    Attributes:
        weight (torch.Tensor): Learnable weight matrix of shape (out_spatial, in_spatial)
                               with enforced orthogonality via `nn.utils.parametrizations.orthogonal`.
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

        # Initialize learnable weight and apply orthogonal parametrization
        weight = torch.empty(out_spatial, in_spatial, **factory_kwargs)
        self.weight = nn.Parameter(weight)
        self.reset_parameters()
        orthogonal(self)  # Registers orthogonal parametrization on `self.weight`

    def reset_parameters(self):
        """
        Initialize the weight matrix with orthogonal initialization.
        """
        init.orthogonal_(self.weight)

    def forward(self, x):
        """
        Apply the bilinear map: W @ X @ W^T.

        Args:
            x (torch.Tensor): Input SPD matrix of shape (*, in_spatial, in_spatial).

        Returns:
            torch.Tensor: Transformed SPD matrix of shape (*, out_spatial, out_spatial).
        """
        return bilinear(x, self.weight)
