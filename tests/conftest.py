import pytest
import torch

from spdnet.parameters import SemiOrthogonalParameter

from .toy_data import make_blobs, make_spd_matrix

# Global settings
BATCH_SIZE = 8
N = 16
C = 3

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


@pytest.fixture(scope="session")
def blob():
    r"""
    Generates synthetic data blobs for clustering.

    This fixture creates a dataset of 2D Gaussian blobs, which can be used for testing clustering algorithms.
    The blobs are generated with 3 centers and a standard deviation of 1.0.

    Returns:
        torch.Tensor: Tensor of shape `(n_samples, n_features)` containing the generated blobs.
        torch.Tensor: Tensor of shape `(n_samples,)` containing the labels for each blob.
    """
    return make_blobs(BATCH_SIZE, N)


@pytest.fixture(scope="session")
def semi_orthogonal_network():
    r"""
    Creates a simple semi-orthogonal network for testing.

    Returns:
        torch.nn.Module: A network with semi-orthogonal weight in the first layer.
    """
    network = torch.nn.Sequential(torch.nn.Linear(N, N // 2, bias=False), torch.nn.ReLU(), torch.nn.Linear(N // 2, C))

    ortho_weight = torch.nn.init.orthogonal_(torch.empty_like(network[0].weight))
    semi_param = SemiOrthogonalParameter(ortho_weight)

    # Replace and re-register
    network[0].register_parameter("weight", semi_param)

    return network
