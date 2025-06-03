import torch
from torch import nn

from ..functions import sym_mat_pow
from ..metrics import airm_distance, airm_geodesic, airm_parallel_transport, karcher_flow

__all__ = ["RiemannianBatchNorm"]


def riemannian_batch_norm(x, mean, std, scale, eps):
    """
    Applies Riemannian batch normalization to SPD matrices.

    Args:
        x (torch.Tensor): Input tensor of shape (..., n, n), containing SPD matrices.
        mean (torch.Tensor): Mean SPD matrix of shape (..., n, n).
        std (torch.Tensor): Scalar standard deviation.
        scale (torch.Tensor): Learnable scaling factor.
        eps (float): Small epsilon for numerical stability.

    Returns:
        torch.Tensor: Normalized SPD matrices of shape (..., n, n).
    """
    x = airm_parallel_transport(x, mean)
    scale = scale / (std + eps)
    return sym_mat_pow(x, scale)


class RiemannianBatchNorm(nn.Module):
    """
    Riemannian Batch Normalization for SPD matrices using affine-invariant geometry.

    Inspired by:
    - "Riemannian Batch Normalization for SPD Neural Networks" (Huang & Van Gool, NeurIPS 2019)
    - "SPD Domain-Specific Batch Normalization" (Yger et al., NeurIPS 2022)

    Args:
        num_spatial (int): Spatial dimension (size) of the SPD matrices.
        karcher_flow_steps (int): Number of iterations for computing the Karcher mean.
        momentum (float): Momentum for updating running statistics.
        eps (float): Small value added to denominator for numerical stability.
        device (torch.device, optional): Device placement.
        dtype (torch.dtype, optional): Data type.

    Shape:
        - Input: `(*, num_spatial, num_spatial)`
        - Output: `(*, num_spatial, num_spatial)`
    """

    __constants__ = ["num_spatial", "karcher_flow_steps", "momentum", "eps"]
    num_spatial: int
    karcher_flow_steps: int
    momentum: float
    eps: float

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._update_and_fetch_stats(x)
        return riemannian_batch_norm(x, mean, std, self.scale, self.eps)

    def reset_running_stats(self) -> None:
        """
        Resets the running mean to the identity matrix and running variance to 1.
        """
        self.running_mean.copy_(
            torch.eye(self.num_spatial, device=self.running_mean.device, dtype=self.running_mean.dtype)
        )
        self.running_var.fill_(1.0)

    @torch.no_grad()
    def _update_and_fetch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates and retrieves the mean and standard deviation (variance) of SPD matrices.

        During training, computes the batch Karcher mean and affine-invariant variance.
        During evaluation, returns stored running statistics.

        Args:
            x (torch.Tensor): Batch of SPD matrices of shape (..., num_spatial, num_spatial).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean SPD matrix and scalar standard deviation.
        """
        if not self.training:
            return self.running_mean, torch.sqrt(self.running_var)

        mean = karcher_flow(x, self.karcher_flow_steps)
        var = airm_distance(x, mean).pow(2).mean(dim=0)

        # Update running statistics
        self.running_mean.copy_(airm_geodesic(self.running_mean, mean, self.momentum))
        self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)

        return mean, torch.sqrt(var)
