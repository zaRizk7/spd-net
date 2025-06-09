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
    r"""
    Riemannian Batch Normalization (RBN) for symmetric positive-definite (SPD) matrices.

    This layer generalizes classical batch normalization to the Riemannian manifold of SPD matrices.
    It performs the following steps:
        1. Computes the Riemannian batch mean via Karcher flow.
        2. Estimates batch dispersion via the squared Riemannian distance.
        3. Applies parallel transport to the identity, followed by normalization using a learnable
           scalar and the batch standard deviation.

    .. note::
        This implementation **does not include** the learnable bias term proposed in
        Brooks et al. (NeurIPS 2019), which re-centers the normalized output to a learnable
        SPD matrix. The identity matrix is always used as the re-centering base point here,
        as done in Kobler et al. (NeurIPS 2022). Introducing a bias term would require
        Riemannian-specific optimization or reparametrization techniques.

    Args:
        num_spatial (int): Spatial dimension of input SPD matrices (i.e., n for n×n matrices).
        karcher_flow_steps (int, optional): Number of iterations to compute the batch mean via Karcher flow. Default: 1.
        metric (str, optional): Riemannian metric to use; either ``"airm"`` or ``"lem"``. Default: ``"airm"``.
        momentum (float, optional): Momentum factor for the exponential moving average (EMA) of running statistics. Default: 0.1.
        eps (float, optional): Small constant for numerical stability. Default: 1e-5.
        device (torch.device, optional): Device to place the module's parameters and buffers.
        dtype (torch.dtype, optional): Data type of the module's parameters.

    Shape:
        - Input: :math:`(*, n, n)` where `n = num_spatial`
        - Output: :math:`(*, n, n)`

    Attributes:
        scale (torch.Tensor): Learnable scalar used to re-scale normalized matrices.
        running_mean (torch.Tensor): Running Fréchet mean of shape `(n, n)`, initialized to the identity matrix.
        running_var (torch.Tensor): Running variance (scalar), initialized to 1.

    References:
        - Brooks et al., NeurIPS 2019. *Riemannian Batch Normalization for SPD Neural Networks*.
        - Kobler et al., NeurIPS 2022. *SPD Domain-Specific Batch Normalization to Crack Interpretable Unsupervised Domain Adaptation in EEG*.
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
