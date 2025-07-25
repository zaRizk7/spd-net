import torch

__all__ = ["loewner", "symmetrize", "skew"]


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
    tol = torch.finfo(eigvals.dtype).eps

    # (..., n, 1) and (..., 1, n) to broadcast differences
    eig_i = eigvals[..., :, None]
    eig_j = eigvals[..., None, :]
    f_i = f_eigvals[..., :, None]
    f_j = f_eigvals[..., None, :]

    # Difference matrices
    delta_eig = eig_i - eig_j
    delta_f = f_i - f_j

    # Mask where eigenvalues are close enough to be considered equal
    near_diag = torch.abs(delta_eig) < tol

    # Initialize Loewner matrix with safe division
    loewner_matrix = torch.where(near_diag, torch.zeros_like(delta_f), delta_f / delta_eig)

    # Manually set diagonal to df(λ)
    diag_idx = torch.arange(eigvals.shape[-1], device=eigvals.device)
    loewner_matrix[..., diag_idx, diag_idx] = df_eigvals

    return loewner_matrix


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    r"""
    Symmetrizes a tensor by averaging it with its transpose.

    Args:
        x (torch.Tensor): Input tensor of shape `(..., n, n)`.

    Returns:
        torch.Tensor: Symmetrized tensor of shape `(..., n, n)`.
    """
    return (x + x.mT) / 2


def skew(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the skew-symmetric part of a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape `(..., n, n)`.

    Returns:
        torch.Tensor: Skew-symmetric part of the tensor, shape `(..., n, n)`.
    """
    return (x - x.mT) / 2
