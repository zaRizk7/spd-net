import torch

from .affine_invariant import *
from .log_euclidean import *

# Supported metrics:
# - "airm": Affine-Invariant Riemannian Metric
# - "lem": Log-Euclidean Metric

__all__ = ["distance", "geodesic", "log_map", "exp_map", "parallel_transport", "frechet_mean", "karcher_flow"]


def distance(x: torch.Tensor, z: torch.Tensor | None = None, metric: str = "airm") -> torch.Tensor:
    r"""
    Compute the Riemannian distance between SPD matrices `x` and `z` under the specified metric.

    Supported metrics:
        - "airm": d(x, z) = ||log(z^{-1/2} x z^{-1/2})||_F
        - "lem":  d(x, z) = ||log(x) - log(z)||_F

    If `z` is None, the identity matrix is assumed.

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Reference SPD matrix. Defaults to identity.
        metric (str): Metric to use, either "airm" or "lem".

    Returns:
        torch.Tensor: Distance tensor of shape `(...)`.
    """
    if metric == "airm":
        return airm_distance(x, z)
    elif metric == "lem":
        return lem_distance(x, z)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def geodesic(x: torch.Tensor, z: torch.Tensor | None = None, p: float = 0.5, metric: str = "airm") -> torch.Tensor:
    r"""
    Compute a point along the geodesic between SPD matrices `x` and `z`.

    Supported metrics:
        - "airm": γ(p) = z^{1/2} (z^{-1/2} x z^{-1/2})^p z^{1/2}
        - "lem":  γ(p) = exp[(1 - p) log(z) + p log(x)]

    If `z` is None, the identity is assumed.

    Args:
        x (torch.Tensor): Target SPD matrix `(..., N, N)`.
        z (torch.Tensor, optional): Reference SPD matrix. Defaults to identity.
        p (float): Interpolation parameter in `[0, 1]`.
        metric (str): Metric to use.

    Returns:
        torch.Tensor: Interpolated SPD matrix `(..., N, N)`.
    """
    if metric == "airm":
        return airm_geodesic(x, z, p)
    elif metric == "lem":
        return lem_geodesic(x, z, p)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def log_map(x: torch.Tensor, z: torch.Tensor | None = None, metric: str = "airm") -> torch.Tensor:
    r"""
    Compute the Riemannian logarithm map at base point `z` for SPD matrix `x`.

    Supported metrics:
        - "airm": log_z(x) = z^{1/2} log(z^{-1/2} x z^{-1/2}) z^{1/2}
        - "lem":  log_z(x) = log(x) - log(z)

    If `z` is None, identity is assumed.

    Args:
        x (torch.Tensor): SPD matrix `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.
        metric (str): Metric to use.

    Returns:
        torch.Tensor: Tangent vector at `z`, shape `(..., N, N)`.
    """
    if metric == "airm":
        return airm_log(x, z)
    elif metric == "lem":
        return lem_log(x, z)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def exp_map(x: torch.Tensor, z: torch.Tensor | None = None, metric: str = "airm") -> torch.Tensor:
    r"""
    Compute the Riemannian exponential map at base point `z` for a tangent vector `x`.

    Supported metrics:
        - "airm": exp_z(x) = z^{1/2} exp(z^{-1/2} x z^{-1/2}) z^{1/2}
        - "lem":  exp_z(x) = exp(log(z) + x)

    If `z` is None, identity is assumed.

    Args:
        x (torch.Tensor): Tangent vector `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.
        metric (str): Metric to use.

    Returns:
        torch.Tensor: SPD matrix `(..., N, N)`.
    """
    if metric == "airm":
        return airm_exp(x, z)
    elif metric == "lem":
        return lem_exp(x, z)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def parallel_transport(
    x: torch.Tensor, z: torch.Tensor | None = None, s: torch.Tensor | None = None, metric: str = "airm"
) -> torch.Tensor:
    r"""
    Perform parallel transport of tangent vector `x` from base point `z` to point `s`.

    Supported metrics:
        - "airm": PT_{z→s}(x) = (z^{-1} s)^{1/2} x (z^{-1} s)^{1/2}
        - "lem":  PT_{z→s}(x) = x  (trivial transport due to flat geometry)

    Args:
        x (torch.Tensor): Tangent vector at `z` of shape `(..., N, N)`.
        z (torch.Tensor, optional): Source SPD matrix. Defaults to identity.
        s (torch.Tensor, optional): Target SPD matrix. Defaults to identity.
        metric (str): Metric to use.

    Returns:
        torch.Tensor: Transported tangent vector at `s`, shape `(..., N, N)`.
    """
    if metric == "airm":
        return airm_parallel_transport(x, z, s)
    elif metric == "lem":
        return lem_parallel_transport(x, z, s)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def frechet_mean(*x: torch.Tensor, z: torch.Tensor | None = None, metric: str = "airm") -> torch.Tensor:
    r"""
    Compute the Fréchet mean (Karcher mean) of SPD matrices under a Riemannian metric.

    General form:
        μ = exp_z(mean_i(log_z(x_i)))

    If `z` is None, the identity is used:
        μ = exp(mean_i(log(x_i)))

    Args:
        *x (torch.Tensor): Variadic SPD matrices, each of shape `(..., N, N)`.
        z (torch.Tensor, optional): Base point. Defaults to identity.
        metric (str): Metric to use.

    Returns:
        torch.Tensor: Fréchet mean of shape `(..., N, N)`.
    """
    x = torch.stack(x, 0)
    x = log_map(x, z, metric)
    x = torch.mean(x, 0)
    return exp_map(x, z, metric)


def karcher_flow(x: torch.Tensor, steps: int = 1, metric: str = "airm") -> torch.Tensor:
    r"""
    Iteratively estimate the Karcher mean using Riemannian gradient descent.

    The update rule is:
        μ ← exp_μ(mean_i(log_μ(x_i)))

    For the Log-Euclidean metric, a single closed-form Fréchet mean is returned.

    Args:
        x (torch.Tensor): Input SPD matrices of shape `(m, N, N)`.
        steps (int): Number of gradient descent steps.
        metric (str): Metric to use.

    Returns:
        torch.Tensor: Estimated Karcher mean of shape `(N, N)`.
    """
    if metric == "lem":
        return frechet_mean(*x, metric="lem")

    # Initialize with a random point from the batch
    i = torch.randint(x.shape[0], size=(1,)).item()
    mu = x[i]

    for _ in range(steps):
        mu = frechet_mean(*x, z=mu, metric=metric)

    return mu
