import torch

from .autograd import *
from .inner import *
from .linalg import *


def map_fn(x, z, f, **kwargs):
    """
    Computes z^{1/2} * f(z^{-1/2} * x * z^{-1/2}) * z^{1/2}, mapping the SPD matrix x
    to the tangent space at the reference point z using the function fn.

    Args:
        x (torch.Tensor): Input tensor of shape (..., n, n) where n is the spatial dimension.
        z (torch.Tensor): Reference point for the mapping with shape (..., n, n).
        f (callable): Function to apply to the SPD matrix.
        **kwargs: Additional keyword arguments to pass to the function fn.

    Returns:
        torch.Tensor: Tensor with the mapped SPD matrix.
    """
    z_sqrt = powmap(z, 0.5)
    z_sqrt_inv = powmap(z, -0.5)

    x = bilinear(x, z_sqrt_inv)

    return bilinear(f(x, **kwargs), z_sqrt)


def powmap(x, p=2, z=None):
    """
    Compute the power map of a SPD matrix x raised to the power p, optionally in the tangent space at z.

    Args:
        x (torch.Tensor): SPD matrix (..., n, n).
        p (float): Power to which the matrix is raised.
        z (torch.Tensor, optional): Reference point for the power map with shape (..., n, n).
            If None, the power map is computed directly on x.

    Returns:
        torch.Tensor: Power map of the SPD matrix.
    """
    if z is None:
        return SymmetricMatrixPower.apply(x, p)
    return map_fn(x, z, powmap, p=p)


def recmap(x, z=None, eps=1e-5):
    """
    Rectify a SPD matrix x, optionally in the tangent space at z.

    Args:
        x (torch.Tensor): SPD matrix (..., n, n).
        z (torch.Tensor, optional): Reference point for the rectification with shape (..., n, n).
            If None, the rectification is computed directly on x.
        eps (float): Small value to avoid division by zero in variance computation.

    Returns:
        torch.Tensor: Rectified SPD matrix.
    """
    if z is None:
        return SymmetricMatrixRectification.apply(x, eps)
    return map_fn(x, z, recmap, eps=eps)


def sqrtmap(x, z=None):
    """
    Compute the square root of a SPD matrix x, optionally in the tangent space at z.

    Args:
        x (torch.Tensor): SPD matrix (..., n, n).
        z (torch.Tensor, optional): Reference point for the square root with shape (..., n, n).
            If None, the square root is computed directly on x.

    Returns:
        torch.Tensor: Square root of the SPD matrix.
    """
    if z is None:
        return powmap(x, 0.5)
    return map_fn(x, z, sqrtmap)


def invmap(x, z=None):
    """
    Compute the inverse of a SPD matrix x, optionally in the tangent space at z.

    Args:
        x (torch.Tensor): SPD matrix (..., n, n).
        z (torch.Tensor, optional): Reference point for the inverse with shape (..., n, n).
            If None, the inverse is computed directly on x.

    Returns:
        torch.Tensor: Inverse of the SPD matrix.
    """
    if z is None:
        return powmap(x, -1)
    return map_fn(x, z, invmap)


def logmap(x, z=None):
    """
    Compute the logarithm of a SPD matrix x, optionally in the tangent space at z.

    Args:
        x (torch.Tensor): SPD matrix (..., n, n).
        z (torch.Tensor, optional): Reference point for the logarithm with shape (..., n, n).
            If None, the logarithm is computed directly on x.

    Returns:
        torch.Tensor: Logarithm of the SPD matrix.
    """
    if z is None:
        return SymmetricMatrixLogarithm.apply(x)
    return map_fn(x, z, logmap)


def expmap(x, z=None):
    """
    Compute the exponential map of a SPD matrix x, optionally in the tangent space at z.

    Args:
        x (torch.Tensor): SPD matrix (..., n, n).
        z (torch.Tensor, optional): Reference point for the exponential map with shape (..., n, n).
            If None, the exponential map is computed directly on x.

    Returns:
        torch.Tensor: Exponential map of the SPD matrix.
    """
    if z is None:
        return SymmetricMatrixExponential.apply(x)
    return map_fn(x, z, expmap)


def frechet_mean(*x, z=None):
    """
    Compute a single step of the Fréchet mean of a batch of SPD matrices
    assuming the sample weights are uniform.

    There are two behaviors depending on whether `z` is provided:
    - If `z` is provided, the mean is computed in the tangent space at `z`,
      producing an affine-invariant mean.
        mu = exp_z(mean(log_z(x)))
      Note that with affine-invariant mean, this function only applies a single
      step to approximate the Fréchet mean. Please use `karcher_flow` for
      iterative computation of the Fréchet mean.
    - Else, the mean is computed in the log-Euclidean space, obtaining
      the arithmetic mean.
        mu = exp(mean(log(x)))

    Both exp and log are SPD exponential and logarithm maps.

    Args:
        x (torch.Tensor): Input tensor of shape (..., n, n) where n is the spatial dimension.
        z (torch.Tensor, optional): Reference point for the mean with shape (..., n, n).
            If provided, the mean will be computed in the tangent space at this point.

    Returns:
        torch.Tensor: Tensor with the Fréchet mean of the input matrices.
    """
    x = torch.stack(x, 0)
    x = logmap(x, z)
    x = torch.mean(x, 0)
    return expmap(x, z)


def karcher_flow(x, steps=1):
    """
    Compute the Karcher mean of a batch of SPD matrices using the Karcher flow.
    The Karcher mean is computed by iteratively applying the Fréchet mean.

    Args:
        x (torch.Tensor): Input tensor of shape (n, n, n) where n is the spatial dimension.
            The first dimension is the batch size.
        steps (int): Number of iterations for the Karcher flow.

    Returns:
        torch.Tensor: Tensor with the Karcher mean of the input matrices.
    """
    # 1. sample a single point on the manifold as reference point
    index = torch.randint(0, x.shape[0], (1,))[0]
    mu = x[index]

    # 2. compute the Karcher mean
    for _ in range(steps):
        mu = frechet_mean(*x, mu)

    return mu


def affine_invariant_distance(x, z):
    """
    Compute the affine-invariant distance between two SPD matrices x and z.
    The distance is computed as the Frobenius norm of the logarithm of the
    product of the inverse square root of z and x, and the square root of z.
    The formula is:
        d(x, z) = || log(z^{-1/2} * x * z^{-1/2}) ||_F
    Args:
        x (torch.Tensor): SPD matrix (..., n, n).
        z (torch.Tensor): Reference SPD matrix (..., n, n).

    Returns:
        torch.Tensor: Affine-invariant distance between x and z.
    """
    z_sqrt_inv = powmap(z, -0.5)

    d = bilinear(x, z_sqrt_inv)
    d = logmap(d)
    d = torch.einsum("...ij,...ij->...", d, d)
    return torch.sqrt(d)


def parallel_transport(x, z, s=None):
    """
    Compute the parallel transport of a SPD matrix x to the tangent space of z.

    Compute PT_{z->s}(x) = (z^{-1}s)^{1/2} * x * (z^{-1}s)^{1/2}

    Args:
        x (torch.Tensor): SPD matrix (..., n, n).
        z (torch.Tensor):  Reference point for the parallel transport with shape (..., n, n).
        s (torch.Tensor, optional): Target point for the parallel transport with shape (..., n, n).
    If None, s is assumed to be an identity matrix of the same shape as z.

    Returns:
        torch.Tensor: Parallel transported matrix.
    """
    if s is None:
        e = powmap(z, -0.5)
        return bilinear(x, e)

    z_inv = invmap(z)
    e = bdot(z_inv, s)
    e = powmap(e, 0.5)
    return bilinear(x, e)
