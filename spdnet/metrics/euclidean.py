import torch

from ..functions import fro

__all__ = ["euc_distance", "euc_geodesic", "euc_log", "euc_exp", "euc_parallel_transport"]


def euc_distance(x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the Euclidean (Frobenius) distance between SPD matrices `x` and `z`.

        d_EUC(x, z) = ||x - z||_F

    If `z` is None, the identity matrix is assumed:

        d_EUC(x, I) = ||x - I||_F

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Reference SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: Frobenius norm distance, shape `(...)`.
    """
    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    return fro(x - z)


def euc_geodesic(x: torch.Tensor, z: torch.Tensor | None = None, p: float = 0.5) -> torch.Tensor:
    r"""
    Compute a point along the Euclidean geodesic (straight line) from `z` to `x`.

        γ(p) = (1 - p) * z + p * x

    If `z` is None, the identity matrix is assumed:

        γ(p) = (1 - p) * I + p * x

    Args:
        x (torch.Tensor): Target SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.
        p (float): Interpolation parameter in `[0, 1]`.

    Returns:
        torch.Tensor: Interpolated matrix along geodesic, shape `(..., N, N)`.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("Interpolation parameter 'p' must lie in [0, 1].")

    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    return (1.0 - p) * z + p * x


def euc_log(x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the Euclidean logarithm map of `x` at base point `z`.

        log_z(x) = x - z

    If `z` is None, the identity matrix is assumed:

        log_I(x) = x - I

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: Tangent vector at `z`, shape `(..., N, N)`.
    """
    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    return x - z


def euc_exp(v: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the Euclidean exponential map of tangent vector `v` at base point `z`.

        exp_z(v) = z + v

    If `z` is None, the identity matrix is assumed:

        exp_I(v) = I + v

    Args:
        v (torch.Tensor): Tangent vector of shape `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: SPD matrix of shape `(..., N, N)`.
    """
    if z is None:
        z = torch.eye(v.shape[-1], device=v.device, dtype=v.dtype)
    return z + v


def euc_parallel_transport(
    v: torch.Tensor, z: torch.Tensor | None = None, s: torch.Tensor | None = None
) -> torch.Tensor:
    r"""
    Compute the Euclidean parallel transport of tangent vector `v` from `z` to `s`.

    In Euclidean space, parallel transport is identity:

        PT_{z→s}(v) = v

    Args:
        v (torch.Tensor): Tangent vector of shape `(..., N, N)`.
        z (torch.Tensor, optional): Source SPD matrix. Ignored.
        s (torch.Tensor, optional): Target SPD matrix. Ignored.

    Returns:
        torch.Tensor: Unchanged tangent vector `v`, shape `(..., N, N)`.
    """
    return v
