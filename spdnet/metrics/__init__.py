import torch

from .affine_invariant import *
from .log_euclidean import *

# For now, we limit the computation to AIRM and LEM; it can be extended to other metrics later.


def distance(x, z=None, metric="airm"):
    """
    Compute the distance between SPD matrices *x* and *z* using the specified Riemannian metric.

    Supported metrics:
        - "airm": Affine-Invariant Riemannian Metric (AIRM)
            d(x, z) = || log(z^{-1/2} x z^{-1/2}) ||_F
        - "lem": Log-Euclidean Metric (LEM)
            d(x, z) = || log(x) - log(z) ||_F

    If *z* is None, identity is assumed:
        - AIRM: d(x, I) = || log(x) ||_F
        - LEM:  d(x, I) = || log(x) ||_F

    Args:
        x (torch.Tensor): SPD matrix of shape (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix (..., N, N) or (N, N). Defaults to identity.
        metric (str): Metric to use, either "airm" or "lem".

    Returns:
        torch.Tensor: Distance values with shape (...,).
    """
    if metric == "airm":
        return airm_distance(x, z)
    elif metric == "lem":
        return lem_distance(x, z)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def geodesic(x, z=None, p=0.5, metric="airm"):
    """
    Compute a point along the geodesic between SPD matrices *x* and *z* under the specified metric.

    Supported metrics:
        - "airm": γ(p) = z^{1/2} (z^{-1/2} x z^{-1/2})^p z^{1/2}
        - "lem": γ(p) = exp((1 - p) log(z) + p log(x))

    If *z* is None, identity is assumed:
        - AIRM: γ(p) = x^p
        - LEM:  γ(p) = exp(p log(x))

    Args:
        x (torch.Tensor): Target SPD matrix (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix (..., N, N) or (N, N). Defaults to identity.
        p (float): Interpolation parameter in [0, 1].
        metric (str): Metric to use, either "airm" or "lem".

    Returns:
        torch.Tensor: Interpolated SPD matrix (..., N, N).
    """
    if metric == "airm":
        return airm_geodesic(x, z, p)
    elif metric == "lem":
        return lem_geodesic(x, z, p)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def log_map(x, z=None, metric="airm"):
    """
    Compute the Riemannian logarithm map at base point *z* for SPD matrix *x*.

    Supported metrics:
        - "airm": log_z(x) = z^{1/2} log(z^{-1/2} x z^{-1/2}) z^{1/2}
        - "lem":  log_z(x) = log(x) - log(z)

    If *z* is None, identity is assumed:
        - AIRM: log_I(x) = log(x)
        - LEM:  log_I(x) = log(x)

    Args:
        x (torch.Tensor): SPD matrix (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix (..., N, N) or (N, N). Defaults to identity.
        metric (str): Metric to use, either "airm" or "lem".

    Returns:
        torch.Tensor: Tangent vector at z (..., N, N).
    """
    if metric == "airm":
        return airm_log(x, z)
    elif metric == "lem":
        return lem_log(x, z)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def exp_map(x, z=None, metric="airm"):
    """
    Compute the Riemannian exponential map at base point *z* for tangent vector *x*.

    Supported metrics:
        - "airm": exp_z(x) = z^{1/2} exp(z^{-1/2} x z^{-1/2}) z^{1/2}
        - "lem":  exp_z(x) = exp(log(z) + x)

    If *z* is None, identity is assumed:
        - AIRM: exp_I(x) = exp(x)
        - LEM:  exp_I(x) = exp(x)

    Args:
        x (torch.Tensor): Tangent vector (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix (..., N, N) or (N, N). Defaults to identity.
        metric (str): Metric to use, either "airm" or "lem".

    Returns:
        torch.Tensor: SPD matrix (..., N, N).
    """
    if metric == "airm":
        return airm_exp(x, z)
    elif metric == "lem":
        return lem_exp(x, z)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def parallel_transport(x, z=None, s=None, metric="airm"):
    """
    Compute parallel transport of tangent vector *x* from base point *z* to target point *s*.

    Supported metrics:
        - "airm": PT_{z→s}(x) = (z^{-1} s)^{1/2} x (z^{-1} s)^{1/2}
        - "lem":  PT_{z→s}(x) = x  (parallel transport is trivial in the log-Euclidean metric)

    If *z* or *s* is None, identity is assumed.

    Args:
        x (torch.Tensor): Tangent vector at z (..., N, N).
        z (torch.Tensor, optional): Source SPD matrix (..., N, N). Defaults to identity.
        s (torch.Tensor, optional): Target SPD matrix (..., N, N). Defaults to identity.
        metric (str): Metric to use, either "airm" or "lem".

    Returns:
        torch.Tensor: Transported tangent vector at s (..., N, N).
    """
    if metric == "airm":
        return airm_parallel_transport(x, z, s)
    elif metric == "lem":
        return lem_parallel_transport(x, z, s)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'airm' or 'lem'.")


def frechet_mean(*x, z=None, metric="airm"):
    """
    Compute the Fréchet (Karcher) mean of SPD matrices under the specified Riemannian metric.

    General formulation:
        μ = exp_z(mean_i(log_z(x_i)))

    If *z* is None, identity is used, and simplified to:
        μ = exp(mean_i(log(x_i)))

    Args:
        *x (torch.Tensor): Variadic SPD matrices, each of shape (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix (..., N, N). Defaults to identity.
        metric (str): Metric to use, either "airm" or "lem".

    Returns:
        torch.Tensor: Fréchet mean SPD matrix (..., N, N).
    """
    x = torch.stack(x, 0)
    x = log_map(x, z, metric)
    x = torch.mean(x, 0)
    return exp_map(x, z, metric)


def karcher_flow(x, steps=1, metric="airm"):
    """
    Estimate the Karcher mean via iterative updates under the specified Riemannian metric.

    Starting from a randomly selected point, apply:
        μ ← exp_μ(mean_i(log_μ(x_i)))

    This process converges to the Fréchet mean under mild conditions.
    For log-Euclidean metric, `frechet_mean(x, metric="lem")` will be used instead.

    Args:
        x (torch.Tensor): SPD matrices of shape (m, N, N), where m is the number of matrices.
        steps (int): Number of update iterations (default: 1).
        metric (str): Metric to use, either "airm" or "lem".

    Returns:
        torch.Tensor: Karcher mean SPD matrix (N, N).
    """
    if metric == "lem":
        return frechet_mean(*x, metric="lem")

    i = torch.randint(x.shape[0], size=(1,)).item()
    mu = x[i]
    for _ in range(steps):
        mu = frechet_mean(*x, z=mu, metric=metric)
    return mu
