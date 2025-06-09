import pytest
import torch

from .toy_data import make_spd_matrix

BATCH_SIZE = 1
N = 32

torch.manual_seed(0)


@pytest.fixture(scope="session")
def x():
    return torch.stack([make_spd_matrix(N, dtype=torch.float64).requires_grad_() for _ in range(BATCH_SIZE)], dim=0)


@pytest.fixture(scope="session")
def w():
    w = torch.empty(N // 2, N, dtype=torch.float64)
    torch.nn.init.orthogonal_(w)
    return w.requires_grad_()
