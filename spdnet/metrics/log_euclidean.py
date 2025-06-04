import opt_einsum as oe
import torch

from ..functions import (
    bdot,
    bilinear,
    eig2matrix,
    sym_mat_exp,
    sym_mat_inv,
    sym_mat_log,
    sym_mat_pow,
    sym_mat_sqrt,
    fro,
)


def lem_distance(x, z=None):
    """
    Compute the Log-Euclidean distance between SPD matrices *x* and *z*.

    Original LEM formulation:
        d_lem(x, z) = ‖log(x) - log(z)‖_F

    If *z* is None (identity is assumed):
        d_lem(x, I) = ‖log(x)‖_F

    Args:
        x (torch.Tensor): SPD matrix of shape (..., N, N).
        z (torch.Tensor, optional): SPD matrix (..., N, N) or (N, N). If None, identity is assumed.

    Returns:
        torch.Tensor: Frobenius norm distance value(s), shape (...,).
    """
    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    d = lem_log(x, z)
    return fro(d)


def lem_geodesic(x, z=None, p=0.5):
    """
    Compute a point p-fraction along the Log-Euclidean geodesic from *z* to *x*.

    Original LEM formulation:
        γ(p) = exp[(1 − p) log(z) + p log(x)],     where 0 ≤ p ≤ 1

    Special cases:
        p = 0        → γ(0) = z
        p = 1        → γ(1) = x
        z is None    → γ(p) = exp(p log(x))

    Args:
        x (torch.Tensor): Target SPD matrix (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix (..., N, N) or (N, N). If None, identity is assumed.
        p (float, optional): Interpolation parameter in [0, 1]. Default is 0.5.

    Returns:
        torch.Tensor: Interpolated SPD matrix (..., N, N).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("Interpolation parameter 'p' must lie in [0, 1].")

    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)

    g = (1.0 - p) * sym_mat_log(z) + p * sym_mat_log(x)
    return sym_mat_exp(g)


def lem_log(x, z=None):
    """
    Compute the Log-Euclidean Riemannian logarithm map of *x* at base point *z*.

    Original LEM formula:
        log_z(x) = log(x) - log(z)

    If *z* is None (identity is assumed):
        log_I(x) = log(x)

    Args:
        x (torch.Tensor): SPD matrix (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix (..., N, N) or (N, N). If None, identity is assumed.

    Returns:
        torch.Tensor: Tangent vector at *z*, shape (..., N, N).
    """
    if z is None:
        z = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    return sym_mat_log(x) - sym_mat_log(z)


def lem_exp(v, z=None):
    """
    Compute the Log-Euclidean Riemannian exponential map of tangent vector *v* at base point *z*.

    Original LEM formula:
        exp_z(v) = exp(log(z) + v)

    If *z* is None (identity is assumed):
        exp_I(v) = exp(v)

    Args:
        v (torch.Tensor): Tangent vector (..., N, N) (symmetric, not necessarily SPD).
        z (torch.Tensor, optional): Reference SPD matrix (..., N, N) or (N, N). If None, identity is assumed.

    Returns:
        torch.Tensor: SPD matrix resulting from the exponential map (..., N, N).
    """
    if z is None:
        return sym_mat_exp(v)
    return sym_mat_exp(sym_mat_log(z) + v)


def lem_parallel_transport(v, z=None, s=None):
    """
    Compute the parallel transport of a tangent vector *v* from *z* to *s* under the Log-Euclidean metric.

    The Log-Euclidean metric induces a flat (Euclidean) geometry on the space of symmetric matrices.
    Hence, parallel transport is identity — the tangent vector remains unchanged.

    Args:
        v (torch.Tensor): Tangent vector (..., N, N).
        z (torch.Tensor, optional): Source SPD matrix (ignored).
        s (torch.Tensor, optional): Target SPD matrix (ignored).

    Returns:
        torch.Tensor: The original tangent vector *v*.
    """
    return v
