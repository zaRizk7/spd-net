import torch

__all__ = ["loewner"]


def loewner(eigvals: torch.Tensor, f_eigvals: torch.Tensor, df_eigvals: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the Loewner matrix for a scalar function `f` applied to a symmetric matrix.

    For eigenvalues `λ`, function values `f(λ)`, and derivatives `f'(λ)`, the Loewner matrix `L` is defined as:

        L_{ij} = (f(λ_i) - f(λ_j)) / (λ_i - λ_j),    if i ≠ j
        L_{ii} = f'(λ_i),                            if i == j

    This matrix is used to compute the gradient of spectral functions (e.g., log, exp, power)
    over SPD matrices via backpropagation.

    Args:
        eigvals (torch.Tensor): Eigenvalues of shape `(..., n)`.
        f_eigvals (torch.Tensor): Function values at eigenvalues `f(λ)`, shape `(..., n)`.
        df_eigvals (torch.Tensor): Derivatives at eigenvalues `f'(λ)`, shape `(..., n)`.

    Returns:
        torch.Tensor: Loewner matrix of shape `(..., n, n)`.
    """
    eps = torch.finfo(eigvals.dtype).eps

    # Compute f(λ_i) - f(λ_j)
    delta_f = f_eigvals[..., None, :] - f_eigvals[..., :, None]
    # Compute λ_i - λ_j
    delta_eigvals = eigvals[..., None, :] - eigvals[..., :, None]

    # Mask for diagonal elements (where i == j)
    is_diagonal = torch.abs(delta_eigvals) < eps

    # Safe division for off-diagonal elements
    loewner_matrix = delta_f / (delta_eigvals + eps)
    # In-place assignment for diagonal elements
    torch.where(is_diagonal, df_eigvals[..., None], loewner_matrix, out=loewner_matrix)

    # Prevent NaNs in the Loewner matrix
    torch.nan_to_num_(loewner_matrix)

    return loewner_matrix
