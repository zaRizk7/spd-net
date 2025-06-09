import torch
from torch import nn

from ..functions import sym_mat_pow
from ..metrics import distance, geodesic, karcher_flow, parallel_transport

__all__ = ["RiemannianBatchNorm"]


def riemannian_batch_norm(x, mean, std, scale=1, eps=1e-5, metric="airm"):
    """
    Apply Riemannian batch normalization to a batch of SPD matrices.

    This function parallel transports each SPD matrix to the identity tangent space,
    then rescales using the standard deviation and a learnable scalar scale.

    Args:
        x (Tensor): Input SPD matrices of shape (..., n, n).
        mean (Tensor): Batch Fréchet mean of shape (..., n, n).
        std (Tensor): Scalar standard deviation of the batch.
        scale (Tensor): Learnable scale parameter (default: 1).
        eps (float): Small value to prevent division by zero.
        metric (str): Riemannian metric to use ("airm" or "lem").

    Returns:
        Tensor: Normalized SPD matrices of shape (..., n, n).
    """
    x = parallel_transport(x, mean, metric=metric)
    scale = scale / (std + eps)
    return sym_mat_pow(x, scale)


class RiemannianBatchNorm(nn.Module):
    """
    Riemannian Batch Normalization (RBN) for SPD matrices.

    This layer generalizes batch normalization to the SPD manifold by:
        1. Computing the batch Fréchet mean via Karcher flow.
        2. Estimating the batch variance via squared Riemannian distances.
        3. Applying parallel transport to the identity and scaling.

    References:
        - Brooks et al., NeurIPS 2019
        - Kobler et al., NeurIPS 2022

    Args:
        num_spatial (int): Spatial dimension of input SPD matrices (n × n).
        karcher_flow_steps (int): Number of iterations for mean estimation (default: 1).
        metric (str): Riemannian metric to use ("airm" or "lem").
        momentum (float): EMA momentum for running statistics (default: 0.1).
        eps (float): Stability constant to avoid divide-by-zero (default: 1e-5).
        device (torch.device, optional): Device to place parameters on.
        dtype (torch.dtype, optional): Data type for parameters.
    """

    __constants__ = ["num_spatial", "karcher_flow_steps", "metric", "momentum", "eps"]
    num_spatial: int
    karcher_flow_steps: int
    metric: str
    momentum: float
    eps: float

    def __init__(
        self, num_spatial, karcher_flow_steps=1, metric="airm", momentum=0.1, eps=1e-5, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_spatial = num_spatial
        self.karcher_flow_steps = karcher_flow_steps
        self.metric = metric
        self.momentum = momentum
        self.eps = eps

        self.scale = nn.Parameter(torch.ones(1, **factory_kwargs))
        self.register_buffer("running_mean", torch.eye(num_spatial, **factory_kwargs))
        self.register_buffer("running_var", torch.ones(1, **factory_kwargs))

    def forward(self, x):
        """
        Forward pass of Riemannian BatchNorm.

        Args:
            x (Tensor): Batch of SPD matrices of shape (..., num_spatial, num_spatial).

        Returns:
            Tensor: Normalized SPD matrices.
        """
        mean, std = self._update_and_fetch_stats(x)
        return riemannian_batch_norm(x, mean, std, self.scale, self.eps, self.metric)

    def reset_running_stats(self):
        """
        Reset the running statistics to default:
            - running_mean ← identity
            - running_var ← 1
        """
        self.running_mean.copy_(
            torch.eye(self.num_spatial, device=self.running_mean.device, dtype=self.running_mean.dtype)
        )
        self.running_var.fill_(1.0)

    @torch.no_grad()
    def _update_and_fetch_stats(self, x):
        """
        Update (if training) and fetch the batch statistics.

        During training:
            - Computes batch mean and variance.
            - Updates EMA of running statistics via geodesic interpolation.

        During evaluation:
            - Uses stored running_mean and running_var.

        Args:
            x (Tensor): Batch of SPD matrices of shape (..., num_spatial, num_spatial).

        Returns:
            Tuple[Tensor, Tensor]: (mean SPD matrix, scalar std dev)
        """
        if not self.training:
            return self.running_mean, torch.sqrt(self.running_var)

        mean = karcher_flow(x, self.karcher_flow_steps, metric=self.metric)
        var = distance(x, mean, metric=self.metric).square().mean(dim=0)

        self.running_mean.copy_(geodesic(self.running_mean, mean, self.momentum, self.metric))
        self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)

        return mean, torch.sqrt(var)
