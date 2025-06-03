import torch
from torch import nn

from ..functions import affine_invariant_distance, karcher_flow, powmap, parallel_transport

__all__ = ["RiemannianBatchNorm"]


def riemannian_batch_norm(x, mean, std, scale, eps):
    """
    Riemannian batch normalization for SPD matrices.

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., n, n).
        mean (torch.Tensor): Mean SPD matrix of shape (..., n, n).
        std (torch.Tensor): Standard deviation of the SPD matrices.
        scale (torch.Tensor): Scaling factor for normalization.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Normalized SPD matrix.
    """
    x = parallel_transport(x, mean)
    scale = scale / (std + eps)
    return powmap(x, scale)


class RiemannianBatchNorm(nn.Module):
    """
    Riemannian Batch Normalization for SPD matrices. Implementation is based from both
    "Riemannian batch normalization for SPD neural networks" (NeurIPS 2019) and "SPD
    domain-specific batch normalization to crack interpretable unsupervised domain
    adaptation in EEG" (NeurIPS 2022)

    Args:
        num_spatial (int): Spatial dimension of the SPD matrices.
        karcher_flow_steps (int): Number of steps for Karcher flow to compute the mean.
        momentum (float): Momentum for running statistics update.
        eps (float): Small value to avoid division by zero in variance computation.

    Shape:
        - input: (*, num_spatial, num_spatial)
        - output: (*, num_spatial, num_spatial)

    Attributes:
        - scale (torch.nn.Parameter): Learnable scaling factor for normalization.
        - running_mean (torch.Tensor): Running mean of the SPD matrices.
        - running_var (torch.Tensor): Running variance of the SPD matrices.
        - eps (torch.Tensor): Small value for numerical stability.
    """

    def __init__(self, num_spatial, karcher_flow_steps=1, momentum=0.1, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_spatial = num_spatial
        self.karcher_flow_steps = karcher_flow_steps
        self.momentum = momentum
        self.eps = eps

        self.scale = nn.Parameter(torch.ones(1, **factory_kwargs))
        self.register_buffer("running_mean", torch.eye(num_spatial, **factory_kwargs))
        self.register_buffer("running_var", torch.ones(1, **factory_kwargs))

    def forward(self, x):
        mean, std = self.update_and_fetch_stats(x)
        return riemannian_batch_norm(x, mean, std, self.scale, self.eps)

    def reset_running_stats(self):
        """Reset the running statistics to their initial values."""
        self.running_mean.eye_(self.num_spatial)
        self.running_var.fill_(1.0)

    @torch.no_grad()
    def update_and_fetch_stats(self, x):
        """Update running statistics and return the current mean and variance.

        Args:
            x (torch.Tensor): Input SPD matrix of shape (..., num_spatial, num_spatial).

        Returns:
            tuple (torch.Tensor, torch.Tensor): Mean and variance of the SPD matrices.
        """
        if not self.training:
            return self.running_mean, torch.sqrt(self.running_var)

        mean = karcher_flow(x, self.karcher_flow_steps)
        var = affine_invariant_distance(x, mean).mean(0)

        self.running_mean.copy_(powmap(self.running_mean, self.momentum, mean))
        self.running_var.copy_(self.running_var * (1 - self.momentum) + var * self.momentum)

        return mean, torch.sqrt(var)
