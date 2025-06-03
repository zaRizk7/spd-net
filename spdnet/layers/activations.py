import torch
from torch import linalg as la
from torch import nn

from ..functions import expmap, logmap, recmap

__all__ = ["EigenActivation"]


class EigenActivation(nn.Module):
    """
    Applies an activation function to the eigenvalues of a SPD matrix.

    Args:
        activation (str): The activation function to apply to the eigenvalues. Options are 'rectify', 'log', or 'exp'.
        eps (float): Small value to ensure numerical stability for 'rectify' activation function.
        device (torch.device, optional): The device to place the module on.
        dtype (torch.dtype, optional): The data type of the module's parameters.
    """

    __constants__ = ["activation", "eps"]
    activation: str
    eps: float

    def __init__(self, activation="rectify", eps=1e-5, device=None, dtype=None):
        if activation not in {"rectify", "log", "exp"}:
            msg = f"activation must be one of 'rectify', 'log', or 'exp'. Got '{activation}'."
            raise ValueError(msg)

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.activation = activation
        self.register_buffer("eps", torch.tensor(eps, **factory_kwargs))

    def forward(self, x):
        if self.activation == "rectify":
            return recmap(x, eps=self.eps)
        elif self.activation == "log":
            return logmap(x)
        return expmap(x)
