import torch
from torch import nn

from ..functions import sym_mat_pow
from ..metrics import distance, geodesic, karcher_flow, parallel_transport
from ..parameters import SPDParameter

__all__ = ["RiemannianBatchNorm"]


def riemannian_batch_norm(x, mean, std, shift=None, scale=1, eps=1e-5, metric="airm"):
    """
    Apply Riemannian batch normalization to a batch of SPD matrices.

    This function parallel transports each SPD matrix to the identity tangent space,
    then rescales using the standard deviation and a learnable scalar scale.

    The order of operations is: Whiten → Scale → Shift, like the Euclidean BN.

    Args:
        x (Tensor): Input SPD matrices of shape (..., n, n).
        mean (Tensor): Batch Fréchet mean of shape (..., n, n).
        std (Tensor): Scalar standard deviation of the batch.
        shift (Tensor, optional): Learnable (nxn SPD) shift parameter (default: None).
        scale (Tensor): Learnable scale parameter (default: 1).
        eps (float): Small value to prevent division by zero.
        metric (str): Riemannian metric to use ("airm", "lem", or "euclidean").

    Returns:
        Tensor: Normalized SPD matrices of shape (..., n, n).
    """
    # 1. Whiten the input matrices by centering
    x = parallel_transport(x, mean, metric=metric)
    # 2. Scale the whitened matrices
    scale = scale / (std + eps)
    x = sym_mat_pow(x, scale)
    # 3. Shift the whitened and scaled matrices if a shift is provided
    return parallel_transport(x, s=shift, metric=metric)


class RiemannianBatchNorm(nn.Module):
    r"""
    Riemannian Batch Normalization (RBN) for symmetric positive-definite (SPD) matrices.

    This layer generalizes classical batch normalization to the Riemannian manifold of SPD matrices.
    It performs the following steps:
        1. Computes the Riemannian batch mean via Karcher flow.
        2. Estimates batch dispersion via the squared Riemannian distance.
        3. Applies parallel transport to the identity, followed by normalization using a learnable
           scalar and the batch standard deviation.

    .. warning::
        When using ``metric="euclidean"``, the geometry is not affine-invariant and
        may produce invalid SPD matrices under extreme extrapolation. Use with care.

    Args:
        num_spatial (int): Spatial dimension of input SPD matrices (i.e., n for n×n matrices).
        karcher_flow_steps (int, optional): Number of iterations to compute the batch mean via Karcher flow. Default: 1.
        metric (str, optional): Riemannian metric to use; one of ``"airm"``, ``"lem"``, or ``"euclidean"``. Default: ``"airm"``.
        momentum (float, optional): Momentum factor for the exponential moving average (EMA) of running statistics. Default: 0.1.
        shift (bool, optional): If True, learn a (nxn SPD) shift parameter (default: True).
        scale (bool, optional): If True, learn a scalar scale parameter (default: True).
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
        self,
        num_spatial,
        karcher_flow_steps=1,
        metric="airm",
        momentum=0.1,
        eps=1e-5,
        shift=True,
        scale=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_spatial = num_spatial
        self.karcher_flow_steps = karcher_flow_steps
        self.metric = metric
        self.momentum = momentum
        self.eps = eps

        if shift:
            self.shift = SPDParameter(torch.eye(num_spatial, **factory_kwargs))
        else:
            self.register_parameter("shift", None)

        if scale:
            self.scale = nn.Parameter(torch.ones(1, **factory_kwargs))
        else:
            self.register_parameter("scale", None)

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
        var = distance(x, mean, metric=self.metric).square().mean(0)

        self.running_mean.copy_(geodesic(self.running_mean, mean, self.momentum, self.metric))
        self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)

        return mean, torch.sqrt(var)
