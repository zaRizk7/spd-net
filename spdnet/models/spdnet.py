from torch import nn

from ..layers import BiMap, EigenActivation, RiemannianBatchNorm

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

        rectify_last (bool, optional):
            Whether to apply a ReEig after the last BiMap.
            Defaults to False.

        use_batch_norm (bool, optional):
            Whether to use Riemannian batch normalization in the encoder.
            Defaults to False.

        device (torch.device, optional):
            Device to place the model parameters on.
            Defaults to None, which uses the default device.

        dtype (torch.dtype, optional):
            Data type for the model parameters.
            Defaults to None, which uses the default data type.
    """

    def __init__(
        self, num_spatials, num_outputs=None, rectify_last=False, use_batch_norm=False, device=None, dtype=None
    ):
        if len(num_spatials) < 2:
            msg = "num_spatials must contain at least two spatial dimensions."
            raise ValueError(msg)

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        for i in range(1, len(num_spatials)):
            in_spatial = num_spatials[i - 1]
            out_spatial = num_spatials[i]

            name = f"bimap_{i:0=2d}"
            layer = BiMap(in_spatial, out_spatial, **factory_kwargs)
            self.add_module(name, layer)

            if use_batch_norm:
                name = f"bn_{i:0=2d}"
                layer = RiemannianBatchNorm(out_spatial, **factory_kwargs)
                self.add_module(name, layer)

            # Check last layer and skip ReEig if it's a subnetwork
            if i == len(num_spatials) - 1 and not rectify_last:
                continue

            name = f"reeig_{i:0=2d}"
            layer = EigenActivation("rectify", **factory_kwargs)
            self.add_module(name, layer)

        if num_outputs is not None:
            output_layer = nn.Sequential()

            name = "logeig"
            layer = EigenActivation("log", **factory_kwargs)
            output_layer.add_module(name, layer)

            name = "flatten"
            layer = nn.Flatten()
            output_layer.add_module(name, layer)

            name = "linear"
            layer = nn.Linear(out_spatial**2, num_outputs, **factory_kwargs)
            output_layer.add_module(name, layer)

            self.add_module("output", output_layer)
