import torch
from torch import nn

from ..functions import sym_mat_pow
from ..metrics import distance, geodesic, karcher_flow, parallel_transport

__all__ = ["RiemannianBatchNorm"]


def riemannian_batch_norm(x, mean, std, scale=1, eps=1e-5, metric="airm"):
    """
    Apply Riemannian batch normalization to a batch of SPD matrices.

    This function first parallel transports each matrix to the identity, then normalizes
    it using the Riemannian standard deviation and a learnable scalar scale parameter.

    Args:
        x (Tensor): Input SPD matrices of shape (..., n, n).
        mean (Tensor): Batch Fréchet mean (SPD matrix) of shape (..., n, n).
        std (Tensor): Scalar standard deviation of batch.
        scale (Tensor): Learnable scale parameter.
        eps (float): Small epsilon for numerical stability.
        metric (str): Riemannian metric ("airm" or "lem").

    Returns:
        Tensor: Normalized SPD matrices (..., n, n).
    """
    x = parallel_transport(x, mean, metric=metric)
    scale = scale / (std + eps)
    return sym_mat_pow(x, scale)


class RiemannianBatchNorm(nn.Module):
    """
    Riemannian Batch Normalization (RBN) for symmetric positive-definite (SPD) matrices.

    RBN extends classical batch normalization to the manifold of SPD matrices by:
    1. Computing the Riemannian Fréchet mean via Karcher flow.
    2. Estimating the dispersion via geodesic distance.
    3. Normalizing through parallel transport and scaling with a matrix power.

    References:
        - Brooks et al., NeurIPS 2019: "Riemannian Batch Normalization for SPD Neural Networks"
        - Kobler et al., NeurIPS 2022: "SPD Domain-Specific Batch Normalization to Crack Interpretable
          Unsupervised Domain Adaptation in EEG"

    This implementation:
        - Omits bias addition (identity is the canonical center).
        - Uses a learnable scale parameter shared across the batch.
        - Maintains running statistics using exponential moving average during training.

    Args:
        num_spatial (int): Size of each SPD matrix (n × n).
        karcher_flow_steps (int): Number of iterations for Karcher flow to estimate the mean.
        metric (str): Metric to use ("airm" or "lem").
        momentum (float): Momentum for updating running statistics.
        eps (float): Small constant added to denominator for numerical stability.
        device (torch.device, optional): Device for parameter/buffer placement.
        dtype (torch.dtype, optional): Data type for parameters and buffers.

    Shape:
        - Input: `(*, num_spatial, num_spatial)`
        - Output: `(*, num_spatial, num_spatial)`

    Attributes:
        - scale (torch.Tensor): Learnable scale parameter.
        - running_mean (torch.Tensor): Running mean of SPD matrices.
        - running_var (torch.Tensor): Running variance (scalar) for normalization.
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
        Apply Riemannian batch normalization.

        Args:
            x (Tensor): Input batch of SPD matrices of shape (..., num_spatial, num_spatial).

        Returns:
            Tensor: Normalized SPD matrices.
        """
        mean, std = self._update_and_fetch_stats(x)
        return riemannian_batch_norm(x, mean, std, self.scale, self.eps, self.metric)

    def reset_running_stats(self):
        """
        Reset running statistics:
            - Mean ← identity matrix
            - Variance ← 1
        """
        self.running_mean.copy_(
            torch.eye(self.num_spatial, device=self.running_mean.device, dtype=self.running_mean.dtype)
        )
        self.running_var.fill_(1.0)

    @torch.no_grad()
    def _update_and_fetch_stats(self, x):
        """
        Compute and return batch mean and standard deviation for input SPD matrices.

        If in training mode:
            - Updates running mean using the exponential moving average along the geodesic.
            - Updates running variance using squared Riemannian distances.

        If in evaluation mode:
            - Returns stored running statistics.

        Args:
            x (Tensor): Input batch of SPD matrices (..., num_spatial, num_spatial).

        Returns:
            Tuple[Tensor, Tensor]: (mean SPD matrix, scalar standard deviation).
        """
        if not self.training:
            return self.running_mean, torch.sqrt(self.running_var)

        mean = karcher_flow(x, self.karcher_flow_steps, metric=self.metric)
        var = distance(x, mean, metric=self.metric).square().mean(dim=0)

        self.running_mean.copy_(geodesic(self.running_mean, mean, self.momentum, self.metric))
        self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)

        return mean, torch.sqrt(var)
