import pytest
import torch

from .toy_data import make_spd_matrix

BATCH_SIZE = 5
N = 4

torch.manual_seed(0)


@pytest.fixture(scope="session")
def x():
    return torch.stack([make_spd_matrix(N, dtype=torch.float64).requires_grad_() for _ in range(BATCH_SIZE)], dim=0)
