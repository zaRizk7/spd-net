from torch import nn

from ..layers import EigenActivation
from ..metrics import geodesic
from .spdnet import SPDNet

__all__ = ["USPDNet"]


class USPDNet(nn.Module):
    r"""
    U-SPDNet: A U-shaped network for SPD matrix learning, as proposed in:

        Wang et al., "U-SPDNet: An SPD manifold learning-based neural network for visual classification",
        Neural Networks, 2023.

    This architecture follows an encoder-decoder structure operating entirely on
    symmetric positive definite (SPD) matrices using SPDNet blocks. Skip connections
    between encoder and decoder layers are fused using the Log-Euclidean Fréchet mean.

    Args:
        num_spatials (list[int] or tuple[int]):
            Sequence of SPD matrix dimensions for each stage.
            Must have at least two elements. The first is the input size.

        num_outputs (int, optional):
            If specified, adds a fully connected classification head after encoding.
            Default is None (no classifier).

        use_batch_norm (bool, optional):
            Whether to include Riemannian batch normalization in each encoder layer.
            Decoder does not use batch norm due to potential issues with ill-conditioned matrices.

        eps (float, optional):
            Clamping value for ReEig activation to ensure positive definiteness.

        device (torch.device, optional):
            Device to place model parameters on. Defaults to the current device.

        dtype (torch.dtype, optional):
            Data type for model parameters. Defaults to current dtype.
    """

    def __init__(self, num_spatials, num_outputs=None, use_batch_norm=False, eps=1e-5, device=None, dtype=None):
        if len(num_spatials) < 2:
            raise ValueError("`num_spatials` must contain at least two spatial dimensions.")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Build encoder using SPDNet blocks
        for i in range(1, len(num_spatials)):
            in_spatial = num_spatials[i - 1]
            out_spatial = num_spatials[i]

            self.encoder.append(
                SPDNet(
                    (in_spatial, out_spatial),
                    rectify_last=i < len(num_spatials) - 1,  # Only rectify intermediate layers
                    use_batch_norm=use_batch_norm,
                    eps=eps,
                    **factory_kwargs,
                )
            )

        # Build decoder in reverse order; skip batch norm for stability
        for i in reversed(range(1, len(num_spatials))):
            in_spatial = num_spatials[i]
            out_spatial = num_spatials[i - 1]

            self.decoder.append(
                SPDNet((in_spatial, out_spatial), rectify_last=i > 1, use_batch_norm=False, **factory_kwargs)
            )

        # Optional classification head (logEig + flatten + linear)
        self.output = None
        if num_outputs is not None:
            out_spatial = num_spatials[-1]
            self.output = nn.Sequential()
            self.output.add_module("logeig", EigenActivation("log"))
            self.output.add_module("flatten", nn.Flatten(-2, -1))
            self.output.add_module("linear", nn.Linear(out_spatial**2, num_outputs))

    def forward(self, x):
        """
        Forward pass through U-SPDNet.

        Args:
            x (torch.Tensor): Input SPD matrices of shape (..., N, N),
                              where N = num_spatials[0].

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                If `num_outputs` is None: returns the decoded SPD matrix.
                Otherwise: returns a tuple of (decoded SPD, classification output).
        """
        zs = []  # Store encoder outputs for skip connections

        # Encode input SPD matrices
        for layer in self.encoder:
            x = layer(x)
            zs.append(x)

        # Decode with skip connections via Fréchet mean fusion
        for layer, z in zip(self.decoder, reversed(zs[:-1])):
            x = geodesic(layer(x), z, metric="lem")
        x = self.decoder[-1](x)

        # Apply classifier head if defined
        if self.output is not None:
            y = self.output(zs[-1])  # Use final encoder output for prediction
            return x, y

        return x
