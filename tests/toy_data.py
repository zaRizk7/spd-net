import torch


def make_spd_matrix(size, device=None, dtype=None):
    r"""
    Generates a random symmetric positive definite (SPD) matrix of shape `(size, size)`.

    This function constructs an SPD matrix by:
        1. Generating a random matrix `A`.
        2. Forming a symmetric matrix `A.T @ A`.
        3. Applying SVD to orthogonalize the result and stabilize its spectrum.
        4. Adding a diagonal matrix with values > 1 to ensure positive definiteness.

    Args:
        size (int): Size of the square matrix (n × n).
        device (torch.device, optional): Device on which to allocate the tensor.
        dtype (torch.dtype, optional): Desired floating point type of returned tensor.

    Returns:
        torch.Tensor: A symmetric positive definite matrix of shape `(size, size)`.
    """
    # Create a random matrix and use SVD to ensure it is positive definite
    factory_kwargs = {"device": device, "dtype": dtype}
    A = torch.rand(size, size, **factory_kwargs)
    U, _, Vt = torch.linalg.svd(A.T @ A)
    A = U @ (1 + torch.diag(torch.rand(size, **factory_kwargs))) @ Vt

    return A
