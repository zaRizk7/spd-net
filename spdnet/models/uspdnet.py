from torch import nn
from ..layers import *
from .spdnet import SPDNet

__all__ = ["USPDNet"]


class USPDNet(nn.Module):
    """U-SPDNet proposed by Wang et al. in "U-SPDNet: An SPD manifold learning-based
    neural network for visual classification" (Neural Networks 2023).

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
            msg = "num_spatials must contain at least two spatial dimensions."
            raise ValueError(msg)

        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(1, len(num_spatials)):
            in_spatial = num_spatials[i - 1]
            out_spatial = num_spatials[i]

            self.encoder.append(SPDNet((in_spatial, out_spatial), rectify_last=i < len(num_spatials) - 1))

        for i in reversed(range(1, len(num_spatials))):
            in_spatial = num_spatials[i]
            out_spatial = num_spatials[i - 1]

            self.decoder.append(SPDNet((in_spatial, out_spatial), rectify_last=i > 1))

        self.output = None
        if num_outputs is not None:
            out_spatial = num_spatials[-1]
            self.output = nn.Sequential()
            self.output.add_module("logeig", EigenActivation("log"))
            self.output.add_module("flatten", nn.Flatten())
            self.output.add_module("linear", nn.Linear(out_spatial**2, num_outputs))

    def forward(self, x):
        # Encode, keep intermediate outputs for skip connections
        zs = []
        for layer in self.encoder:
            x = layer(x)
            zs.append(x)

        # Decode and apply skip connections
        for layer, z in zip(self.decoder, reversed(zs[:-1])):
            x = arithmetic_mean(layer(x), z)
        x = self.decoder[-1](x)

        # Return with prediction if output layer is defined
        if self.output is not None:
            y = self.output(zs[-1])
            return x, y

        return x
