import torch

from ..functions import fro, sym_mat_exp, sym_mat_log

__all__ = ["lem_distance", "lem_geodesic", "lem_log", "lem_exp", "lem_parallel_transport"]


def lem_distance(x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the Log-Euclidean distance between SPD matrices `x` and `z`.

    The Log-Euclidean distance is defined as:

        d_LEM(x, z) = ||log(x) - log(z)||_F

    If `z` is None, the identity matrix is assumed:

        d_LEM(x, I) = ||log(x)||_F

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Reference SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: Frobenius norm distance, shape `(...)`.
    """
    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    d = lem_log(x, z)
    return fro(d)


def lem_geodesic(x: torch.Tensor, z: torch.Tensor | None = None, p: float = 0.5) -> torch.Tensor:
    r"""
    Compute a point along the Log-Euclidean geodesic from `z` to `x`.

    The Log-Euclidean geodesic is defined as:

        γ(p) = exp[(1 - p) log(z) + p log(x)],     where 0 ≤ p ≤ 1

    If `z` is None, the identity matrix is assumed:

        γ(p) = exp(p * log(x))

    Args:
        x (torch.Tensor): Target SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.
        p (float): Interpolation parameter in `[0, 1]`.

    Returns:
        torch.Tensor: Interpolated SPD matrix along geodesic, shape `(..., N, N)`.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("Interpolation parameter 'p' must lie in [0, 1].")

    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)

    g = (1.0 - p) * sym_mat_log(z) + p * sym_mat_log(x)
    return sym_mat_exp(g)


def lem_log(x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the Log-Euclidean logarithm map of `x` at base point `z`.

    The log map is defined as:

        log_z(x) = log(x) - log(z)

    If `z` is None, the identity matrix is assumed:

        log_I(x) = log(x)

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: Tangent vector at `z`, shape `(..., N, N)`.
    """
    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    return sym_mat_log(x) - sym_mat_log(z)


def lem_exp(v: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the Log-Euclidean exponential map of tangent vector `v` at base point `z`.

    The exponential map is defined as:

        exp_z(v) = exp(log(z) + v)

    If `z` is None, the identity matrix is assumed:

        exp_I(v) = exp(v)

    Args:
        v (torch.Tensor): Tangent vector of shape `(..., N, N)` (symmetric).
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: SPD matrix of shape `(..., N, N)`.
    """
    if z is None:
        return sym_mat_exp(v)
    return sym_mat_exp(sym_mat_log(z) + v)


def lem_parallel_transport(
    v: torch.Tensor, z: torch.Tensor | None = None, s: torch.Tensor | None = None
) -> torch.Tensor:
    r"""
    Compute the Log-Euclidean parallel transport of tangent vector `v` from `z` to `s`.

    Under the Log-Euclidean metric, the manifold is flat in log-space, so parallel transport is identity:

        PT_{z→s}(v) = v

    Args:
        v (torch.Tensor): Tangent vector of shape `(..., N, N)`.
        z (torch.Tensor, optional): Source SPD matrix. Ignored.
        s (torch.Tensor, optional): Target SPD matrix. Ignored.

    Returns:
        torch.Tensor: Unchanged tangent vector `v`, shape `(..., N, N)`.
    """
    return v
