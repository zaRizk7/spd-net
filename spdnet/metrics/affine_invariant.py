import torch

from ..functions import (
    bilinear,
    eig2matrix,
    fro,
    sym_mat_exp,
    sym_mat_inv,
    sym_mat_log,
    sym_mat_pow,
    sym_mat_sqrt,
)

__all__ = ["airm_distance", "airm_geodesic", "airm_log", "airm_exp", "airm_parallel_transport"]


def airm_distance(x, z=None):
    """
    Compute the affine-invariant Riemannian distance between SPD matrices x and z.

    Original AIRM definition:
        d(x, z) = || log(z^{-1/2} x z^{-1/2}) ||_F

    If z is None, it is assumed to be the identity matrix:
        d(x, I) = || log(x) ||_F

    Args:
        x (torch.Tensor): Input SPD matrix of shape (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix. If None, uses identity.

    Returns:
        torch.Tensor: Scalar Riemannian distance for each batch (...,).
    """
    if z is None:
        d = sym_mat_log(x)
    else:
        z_inv_sqrt = sym_mat_pow(z, -0.5)
        d = bilinear(x, z_inv_sqrt)
        d = sym_mat_log(d)
    return fro(d)


def airm_geodesic(x, z=None, p=0.5):
    """
    Compute a point along the geodesic curve between SPD matrices x and z under AIRM.

    Original AIRM geodesic:
        γ(p) = z^{1/2} (z^{-1/2} x z^{-1/2})^p z^{1/2}

    If z is None, identity is assumed:
        γ(p) = x^p

    Args:
        x (torch.Tensor): Target SPD matrix (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix. If None, uses identity.
        p (float): Interpolation parameter between 0 and 1.

    Returns:
        torch.Tensor: Interpolated SPD matrix (..., N, N).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("Parameter 'p' must lie in the interval [0, 1].")

    if z is None:
        return sym_mat_pow(x, p)

    z_sqrt = sym_mat_sqrt(z)
    z_inv_sqrt = sym_mat_pow(z, -0.5)

    g = bilinear(x, z_inv_sqrt)
    g = sym_mat_pow(g, p)

    return bilinear(g, z_sqrt)


def airm_log(x, z=None):
    """
    Compute the Riemannian logarithm map at base point z for SPD matrix x.

    Original AIRM log map:
        log_z(x) = z^{1/2} log(z^{-1/2} x z^{-1/2}) z^{1/2}

    If z is None, identity is assumed:
        log_I(x) = log(x)

    Args:
        x (torch.Tensor): SPD matrix (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix. If None, uses identity.

    Returns:
        torch.Tensor: Tangent vector at z (..., N, N).
    """
    if z is None:
        return sym_mat_log(x)

    z_sqrt = sym_mat_sqrt(z)
    z_inv_sqrt = sym_mat_pow(z, -0.5)

    x_log = bilinear(x, z_inv_sqrt)
    x_log = sym_mat_log(x_log)

    return bilinear(x_log, z_sqrt)


def airm_exp(x, z=None):
    """
    Compute the Riemannian exponential map at base point z for a tangent vector x.

    Original AIRM exponential map:
        exp_z(x) = z^{1/2} exp(z^{-1/2} x z^{-1/2}) z^{1/2}

    If z is None, identity is assumed:
        exp_I(x) = exp(x)

    Args:
        x (torch.Tensor): Tangent vector at z (..., N, N).
        z (torch.Tensor, optional): Reference SPD matrix. If None, uses identity.

    Returns:
        torch.Tensor: SPD matrix (..., N, N).
    """
    if z is None:
        return sym_mat_exp(x)

    z_sqrt = sym_mat_sqrt(z)
    z_inv_sqrt = sym_mat_pow(z, -0.5)

    x_exp = bilinear(x, z_inv_sqrt)
    x_exp = sym_mat_exp(x_exp)

    return bilinear(x_exp, z_sqrt)


def airm_parallel_transport(x, z=None, s=None):
    """
    Parallel transport a tangent vector x from T_zM to T_sM under the affine-invariant Riemannian metric.

    Original AIRM parallel transport:
        PT_{z→s}(x) = (z^{-1} s)^{1/2} x (z^{-1} s)^{1/2}

    If z is None (source is identity):
        PT_{I→s}(x) = s^{1/2} x s^{1/2}

    If s is None (target is identity):
        PT_{z→I}(x) = (z^{-1})^{1/2} x (z^{-1})^{1/2}

    If both z and s are None:
        PT_{I→I}(x) = x

    Args:
        x (torch.Tensor): Tangent vector at z (..., N, N).
        z (torch.Tensor, optional): Source SPD matrix (..., N, N). If None, identity is assumed.
        s (torch.Tensor, optional): Target SPD matrix (..., N, N). If None, identity is assumed.

    Returns:
        torch.Tensor: Transported tangent vector at s (..., N, N).
    """
    if z is None and s is None:
        return x
    elif z is None:
        # PT_{I→s}(x) = s^{1/2} x s^{1/2}
        e = sym_mat_sqrt(s)
    elif s is None:
        # PT_{z→I}(x) = (z^{-1})^{1/2} x (z^{-1})^{1/2}
        e = sym_mat_pow(z, -0.5)
    else:
        # PT_{z→s}(x) = (z^{-1} s)^{1/2} x (z^{-1} s)^{1/2}
        z_inv = sym_mat_inv(z)
        e = z_inv @ s
        e = sym_mat_sqrt(e)

    return bilinear(x, e)
