import torch


def loewner_matrix(eigvals, f_eigvals, df_eigvals):
    """
    Compute the Loewner matrix for function f and its derivative df
    on the eigenvalues.

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape (..., n).
        f_eigvals (torch.Tensor): Function values at the eigenvalues of shape (..., n).
        df_eigvals (torch.Tensor): Derivative of the function at the eigenvalues of shape (..., n).

    Returns:
        torch.Tensor: Loewner matrix of shape (..., n, n).
    """
    eps = torch.finfo(eigvals.dtype).eps

    # Expand dimensions for broadcasting
    eigvals_i, eigvals_j = eigvals[..., None, :], eigvals[..., :, None]
    f_eigvals_i, f_eigvals_j = f_eigvals[..., None, :], f_eigvals[..., :, None]

    # Broadcast tensors to ensure they have compatible shapes
    eigvals_i, eigvals_j = torch.broadcast_tensors(eigvals_i, eigvals_j)
    f_eigvals_i, f_eigvals_j = torch.broadcast_tensors(f_eigvals_i, f_eigvals_j)

    # Compute pairwise differences
    delta_eigvals = eigvals_i - eigvals_j
    delta_f_eigvals = f_eigvals_i - f_eigvals_j
    delta_mask = torch.abs(delta_eigvals) < eps

    # Compute the diagonal elements
    df_eigvals = df_eigvals[..., :, None].expand_as(delta_eigvals)

    # Safe division with eps to avoid division by zero
    delta_f_eigvals /= delta_eigvals + eps

    return torch.where(delta_mask, df_eigvals, delta_f_eigvals)
