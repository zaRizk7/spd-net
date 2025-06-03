import torch


def make_spd_matrix(size, device=None, dtype=None):
    """
    Create a random symmetric positive definite matrix of given size.

    Args:
        batch (int): Number of matrices to create.
        size (int): Size of the matrix (n x n).
        device (torch.device, optional): Device to create the tensor on.
        dtype (torch.dtype, optional): Data type of the tensor.

    Returns:
        torch.Tensor: A symmetric positive definite matrix.
    """
    # Create a random matrix and use SVD to ensure it is positive definite
    factory_kwargs = {"device": device, "dtype": dtype}
    A = torch.rand(size, size, **factory_kwargs)
    U, _, Vt = torch.linalg.svd(A.T @ A)
    A = U @ (1 + torch.diag(torch.rand(size, **factory_kwargs))) @ Vt

    return A
