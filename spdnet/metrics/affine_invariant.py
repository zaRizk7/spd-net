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


def airm_distance(x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the affine-invariant Riemannian distance between SPD matrices `x` and `z`.

    The AIRM distance is defined as:

        d(x, z) = || log(z^{-1/2} x z^{-1/2}) ||_F

    If `z` is None, the identity matrix is used:

        d(x, I) = || log(x) ||_F

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Reference SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: AIRM distance of shape `(...)`.
    """
    if z is None:
        d = sym_mat_log(x)
    else:
        z_inv_sqrt = sym_mat_pow(z, -0.5)
        d = bilinear(x, z_inv_sqrt)
        d = sym_mat_log(d)
    return fro(d)


def airm_geodesic(x: torch.Tensor, z: torch.Tensor | None = None, p: float = 0.5) -> torch.Tensor:
    r"""
    Compute a point on the geodesic between SPD matrices `x` and `z` under AIRM.

    The geodesic is defined as:

        γ(p) = z^{1/2} (z^{-1/2} x z^{-1/2})^p z^{1/2}

    If `z` is None, the identity is assumed:

        γ(p) = x^p

    Args:
        x (torch.Tensor): Target SPD matrix `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.
        p (float): Interpolation parameter between `0` and `1`.

    Returns:
        torch.Tensor: Interpolated SPD matrix of shape `(..., N, N)`.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("Parameter 'p' must lie in the interval [0, 1].")

    if z is None:
        return sym_mat_pow(x, p)

    z_sqrt = sym_mat_sqrt(z)
    z_inv_sqrt = sym_mat_pow(z, -0.5)

    # Align x to z's tangent space
    g = bilinear(x, z_inv_sqrt)
    g = sym_mat_pow(g, p)

    return bilinear(g, z_sqrt)


def airm_log(x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the Riemannian logarithm map at base point `z` for SPD matrix `x`.

    The log map is defined as:

        log_z(x) = z^{1/2} log(z^{-1/2} x z^{-1/2}) z^{1/2}

    If `z` is None, the identity is assumed:

        log_I(x) = log(x)

    Args:
        x (torch.Tensor): SPD matrix of shape `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: Tangent vector at `z`, shape `(..., N, N)`.
    """
    if z is None:
        return sym_mat_log(x)

    z_sqrt = sym_mat_sqrt(z)
    z_inv_sqrt = sym_mat_pow(z, -0.5)

    x_log = bilinear(x, z_inv_sqrt)
    x_log = sym_mat_log(x_log)

    return bilinear(x_log, z_sqrt)


def airm_exp(x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the Riemannian exponential map at base point `z` for a tangent vector `x`.

    The exponential map is defined as:

        exp_z(x) = z^{1/2} exp(z^{-1/2} x z^{-1/2}) z^{1/2}

    If `z` is None, the identity is assumed:

        exp_I(x) = exp(x)

    Args:
        x (torch.Tensor): Tangent vector of shape `(..., N, N)`.
        z (torch.Tensor, optional): Base SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: SPD matrix of shape `(..., N, N)`.
    """
    if z is None:
        return sym_mat_exp(x)

    z_sqrt = sym_mat_sqrt(z)
    z_inv_sqrt = sym_mat_pow(z, -0.5)

    x_exp = bilinear(x, z_inv_sqrt)
    x_exp = sym_mat_exp(x_exp)

    return bilinear(x_exp, z_sqrt)


def airm_parallel_transport(
    x: torch.Tensor, z: torch.Tensor | None = None, s: torch.Tensor | None = None
) -> torch.Tensor:
    r"""
    Parallel transport a tangent vector `x` from `T_zM` to `T_sM` under AIRM.

    The parallel transport is defined as:

        PT_{z→s}(x) = (z^{-1} s)^{1/2} x (z^{-1} s)^{1/2}

    Special cases:
        - If `z` is None (source = identity):  PT_{I→s}(x) = s^{1/2} x s^{1/2}
        - If `s` is None (target = identity):  PT_{z→I}(x) = (z^{-1})^{1/2} x (z^{-1})^{1/2}
        - If both are None:                   PT_{I→I}(x) = x

    Args:
        x (torch.Tensor): Tangent vector of shape `(..., N, N)`.
        z (torch.Tensor, optional): Source SPD matrix. Defaults to identity.
        s (torch.Tensor, optional): Target SPD matrix. Defaults to identity.

    Returns:
        torch.Tensor: Transported tangent vector of shape `(..., N, N)`.
    """
    if z is None and s is None:
        return x
    elif z is None:
        # Transport from identity to s
        e = sym_mat_sqrt(s)
    elif s is None:
        # Transport from z to identity
        e = sym_mat_pow(z, -0.5)
    else:
        # General case: transport from z to s
        z_inv = sym_mat_inv(z)
        e = z_inv @ s
        e = sym_mat_sqrt(e)

    return bilinear(x, e)
