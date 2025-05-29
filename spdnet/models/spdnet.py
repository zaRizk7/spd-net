from torch import nn

from ..layers import *

__all__ = ["SPDNet"]


class SPDNet(nn.Sequential):
    """SPDNet proposed by Huang and van Gool in "A Riemannian Network
    for SPD Matrix Learning" (AAAI 2017).

    Args:
        num_spatials (list[int] or tuple[int]):
            List of spatial dimensions for each layer.
            The first element is the input spatial dimension,
            and the last element is the output spatial dimension.

        num_outputs (int, optional):
            Number of outputs for the final linear layer.
            If None, the network will not have a final linear layer.
            Defaults to None.
    """

    def __init__(self, num_spatials, num_outputs=None):
        if len(num_spatials) < 2:
            raise ValueError("num_spatials must contain at least two spatial dimensions.")

        super().__init__()

        for i in range(1, len(num_spatials)):
            in_spatial = num_spatials[i - 1]
            out_spatial = num_spatials[i]

            name = f"bimap_{i:0=2d}"
            layer = BiMap(in_spatial, out_spatial)
            self.add_module(name, layer)

            name = f"reeig_{i:0=2d}"
            layer = EigenActivation("rectify")
            self.add_module(name, layer)

        if num_outputs is not None:
            output_layer = nn.Sequential()

            name = "logeig"
            layer = EigenActivation("log")
            output_layer.add_module(name, layer)

            name = "flatten"
            layer = nn.Flatten()
            output_layer.add_module(name, layer)

            name = "linear"
            layer = nn.Linear(out_spatial**2, num_outputs)
            output_layer.add_module(name, layer)

            self.add_module("output", output_layer)
