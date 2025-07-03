import torch
from torch import nn

from ..functions import sym_mat_exp, sym_mat_log, sym_mat_rec

__all__ = ["EigenActivation"]


class EigenActivation(nn.Module):
    r"""
    Applies an elementwise transformation to the eigenvalues of a symmetric positive definite (SPD) matrix.

    Supported operations:
        - 'rectify':    Applies ReEig to eigenvalues with an epsilon floor for stability.
        - 'log':        Applies logarithm to eigenvalues.
        - 'exp':        Applies exponential to eigenvalues.

    These operations are commonly used in Riemannian neural networks to preserve SPD structure
    after nonlinear activations.

    Args:
        activation (str):
            The activation to apply. Must be one of {'rectify', 'log', 'exp'}.

        eps (float, optional):
            Minimum eigenvalue threshold used for 'rectify' to ensure positive definiteness.
            Default is 1e-5.

        device (torch.device, optional):
            Device for the internal epsilon buffer. Default is current device.

        dtype (torch.dtype, optional):
            Data type for the internal epsilon buffer. Default is current dtype.
    """

    __constants__ = ["activation", "eps"]
    activation: str
    eps: float

    def __init__(self, activation="rectify", eps=1e-5, device=None, dtype=None):
        if activation not in {"rectify", "log", "exp"}:
            raise ValueError(f"activation must be one of 'rectify', 'log', or 'exp'. Got '{activation}'.")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.activation = activation

        # Store epsilon as a buffer to support device/dtype consistency and tracing
        self.register_buffer("eps", torch.tensor(eps, **factory_kwargs))

    def __repr__(self):
        args = [f"activation='{self.activation}'"]
        if self.activation == "rectify":
            args.append(f"eps={self.eps.item():.1e}")
        return f"{self.__class__.__name__}({', '.join(args)})"

    def forward(self, x):
        """
        Forward pass that applies the selected eigenvalue activation function.

        Args:
            x (torch.Tensor): SPD matrix or batch of SPD matrices of shape (..., N, N).

        Returns:
            torch.Tensor: Transformed SPD matrix (same shape as input).
        """
        if self.activation == "rectify":
            return sym_mat_rec(x, self.eps)
        elif self.activation == "log":
            return sym_mat_log(x)
        else:  # 'exp'
            return sym_mat_exp(x)
