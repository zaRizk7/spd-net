import pytest
import torch

from .toy_data import make_spd_matrix

# Global settings
BATCH_SIZE = 8
N = 16

torch.manual_seed(0)


@pytest.fixture(scope="session")
def x() -> torch.Tensor:
    r"""
    Creates a batch of symmetric positive definite (SPD) matrices.

    Each matrix is of shape `(N, N)` and stacked into a batch of size `BATCH_SIZE`.
    The resulting tensor has shape `(BATCH_SIZE, N, N)` and requires gradients.

    Returns:
        torch.Tensor: Batched SPD tensor of shape `(B, N, N)`, dtype float64.
    """
    return torch.stack([make_spd_matrix(N, dtype=torch.float64).requires_grad_() for _ in range(BATCH_SIZE)], dim=0)


@pytest.fixture(scope="session")
def w() -> torch.Tensor:
    r"""
    Creates an orthogonal weight matrix `w` of shape `(N // 2, N)`.

    This is often used as a projection matrix in bilinear transformations.
    The matrix is orthogonally initialized and requires gradients.

    Returns:
        torch.Tensor: Orthogonal matrix of shape `(N // 2, N)`, dtype float64.
    """
    w = torch.empty(N // 2, N, dtype=torch.float64)
    torch.nn.init.orthogonal_(w)
    return w.requires_grad_()
