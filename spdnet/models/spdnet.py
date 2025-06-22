from torch import nn

from ..layers import BiMap, EigenActivation, RiemannianBatchNorm

__all__ = ["SPDNet"]


class SPDNet(nn.Sequential):
    r"""
    SPDNet implementation based on:

        Huang & van Gool, "A Riemannian Network for SPD Matrix Learning", AAAI 2017.

    This network operates on symmetric positive definite (SPD) matrices and
    applies a sequence of bilinear mappings, optional batch normalization, and
    eigenvalue-based non-linearities (e.g., ReEig, LogEig).

    Args:
        num_spatials (list[int] or tuple[int]):
            Sequence of SPD matrix sizes for each layer.
            `num_spatials[0]` is the input size; `num_spatials[-1]` is the final spatial size.

        num_outputs (int, optional):
            Number of outputs for the final linear classifier.
            If None, classification head is omitted. Default is None.

        rectify_last (bool, optional):
            Whether to apply a ReEig (rectifying EigenActivation) after the last BiMap.
            Default is False.

        use_batch_norm (bool, optional):
            If True, apply Riemannian batch normalization after each BiMap.
            Default is False.

        eps (float, optional):
            Clamping value for ReEig activation to ensure positive definiteness.

        trivialize (bool, optional):
            If True, applies trivialization to BiMap weight to enforce orthogonality.

        device (torch.device, optional):
            Device for model parameters. If None, uses default device.

        dtype (torch.dtype, optional):
            Data type for model parameters. If None, uses default dtype.
    """

    def __init__(
        self,
        num_spatials,
        num_outputs=None,
        rectify_last=False,
        use_batch_norm=False,
        eps=1e-5,
        trivialize=False,
        device=None,
        dtype=None,
    ):
        if len(num_spatials) < 2:
            raise ValueError("`num_spatials` must contain at least two elements (input and output sizes).")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Build feature extraction layers
        for i in range(1, len(num_spatials)):
            in_spatial = num_spatials[i - 1]
            out_spatial = num_spatials[i]

            # Add bilinear mapping layer
            name = f"bimap_{i:0=2d}"
            self.add_module(name, BiMap(in_spatial, out_spatial, trivialize, **factory_kwargs))

            # Optional Riemannian batch norm
            if use_batch_norm:
                name = f"bn_{i:0=2d}"
                self.add_module(name, RiemannianBatchNorm(out_spatial, **factory_kwargs))

            # Optional ReEig (rectify eigenvalues) unless it's the final layer and `rectify_last` is False
            if i == len(num_spatials) - 1 and not rectify_last:
                continue

            name = f"reeig_{i:0=2d}"
            self.add_module(name, EigenActivation("rectify", eps, **factory_kwargs))

        # Optionally add prediction head
        if num_outputs is not None:
            output_layer = nn.Sequential()

            output_layer.add_module("logeig", EigenActivation("log", **factory_kwargs))
            output_layer.add_module("flatten", nn.Flatten())
            output_layer.add_module("linear", nn.Linear(out_spatial**2, num_outputs, **factory_kwargs))

            self.add_module("output", output_layer)
