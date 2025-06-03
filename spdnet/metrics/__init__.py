import torch

from .affine_invariant import *

# For now, we limit the computation to AIRM, it can be extended to other metrics later.


def frechet_mean(*x, z=None):
    """
    Compute the Fréchet mean (also known as the Karcher mean) of a collection of SPD matrices
    under the affine-invariant Riemannian metric.

    Original formulation:
        Find the unique SPD matrix mu that minimizes the sum of squared affine-invariant distances:
            mu = argmin_mu Σ_i || log(z^{-1/2} x_i z^{-1/2}) ||_F^2.
        This can be computed (in one update step) as:
            mu = exp_z( mean_i( log_z(x_i) ) ),
        where log_z and exp_z are the Riemannian logarithm and exponential maps at the base point z.

    Alternative simplified computation when z is None:
        The reference point z is taken as the identity matrix, so that:
            log_I(x) = log(x)  and  exp_I(v) = exp(v).
        Then,
            mu = exp( mean_i( log(x_i) ) ).

    Args:
        *x (Tensor): A variable number of SPD matrices, each with shape (..., N, N).
                     They can be provided as individual tensors.
        z (Tensor, optional): The reference SPD matrix for the logarithm and exponential maps.
                              If None, the identity matrix is assumed in the underlying airm_log and airm_exp calls.

    Returns:
        Tensor: The Fréchet mean of the input SPD matrices, an SPD matrix of shape (..., N, N).
    """
    # Stack input matrices along a new first dimension (n, N, N)
    x = torch.stack(x, 0)
    # Map each SPD matrix to the tangent space at z using the Riemannian logarithm.
    # When z is None, the underlying airm_log assumes the identity matrix.
    x = airm_log(x, z)
    # Compute the Euclidean mean in the tangent space.
    x = torch.mean(x, 0)
    # Map the averaged tangent vector back to the manifold using the exponential map.
    return airm_exp(x, z)


def karcher_flow(x, steps=1):
    """
    Compute the Karcher mean of a set of SPD matrices by performing iterative updates,
    known as the Karcher flow, under the affine-invariant Riemannian metric.

    Original formulation:
        Given a set of SPD matrices {x_i}, the Karcher mean mu is the point that minimizes
            Σ_i d^2_AIRM(x_i, mu).
        The Karcher flow iteratively updates an initial guess mu by computing
            mu_new = frechet_mean(x_1, x_2, ..., x_m; z=mu)
        where the Riemannian logarithm and exponential maps are computed at the current mu.

    Alternative simplified computation when an initial reference is not provided:
        A random SPD matrix from x is chosen as the initial mean, which is equivalent to assuming
        an arbitrary starting reference. The iterative update then refines this estimate.

    Args:
        x (Tensor): A tensor of SPD matrices with shape (m, N, N), where m is the number of matrices.
        steps (int): Number of iterative update steps to perform (default is 1).

    Returns:
        Tensor: The Karcher mean of the input SPD matrices, an SPD matrix of shape (N, N).
    """
    # Choose a random initial candidate from the set as the starting reference for the mean.
    i = torch.randint(x.shape[0], size=(1,)).item()
    mu = x[i]
    # Perform the specified number of iterative updates.
    for _ in range(steps):
        # Update mu by computing the Fréchet mean using the current estimate as the base point.
        mu = frechet_mean(*x, z=mu)
    return mu
