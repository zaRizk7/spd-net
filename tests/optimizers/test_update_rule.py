import torch
import torch.nn.functional as F
from spdnet.parameters import SPDParameter
from spdnet.optimizers.update_rule import update_parameter
from copy import deepcopy


def is_semi_orthogonal(W, atol=1e-2):
    n, p = W.shape[-2:]
    if n < p:
        W = W.mT
    WtW = W.T @ W
    I = torch.eye(W.shape[1], dtype=W.dtype, device=W.device)
    return torch.allclose(WtW, I, atol=atol)


def is_spd(M, atol=1e-5):
    return torch.allclose(M, M.T, atol=atol) and torch.all(torch.linalg.eigvalsh(M) > 0)


def test_update_standard(blob, semi_orthogonal_network):
    X, y = blob
    model = semi_orthogonal_network

    logits = model(X)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    W = model[0].weight
    dW = W.grad.clone()
    expected = W.detach() - 0.1 * dW
    update_parameter(W, dW, lr=0.1)

    assert torch.allclose(W.data, expected, atol=1e-6), "Standard update failed."


def test_update_retraction(blob, semi_orthogonal_network):
    X, y = blob
    model = deepcopy(semi_orthogonal_network)

    logits = model(X)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    W = model[0].weight
    dW = W.grad.clone()

    update_parameter(W, dW, lr=0.1, orth_update_rule="retraction")
    model.zero_grad()

    assert is_semi_orthogonal(W.data), "Retraction update did not preserve orthogonality."


def test_update_landing(blob, semi_orthogonal_network):
    X, y = blob
    model = deepcopy(semi_orthogonal_network)

    for _ in range(10):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        W = model[0].weight
        dW = W.grad.clone()

        update_parameter(W, dW, lr=0.1, orth_update_rule="landing", landing=1.0)
        model.zero_grad()

    assert is_semi_orthogonal(W.data), "Landing update did not preserve orthogonality."


def test_update_spd(x):
    A = x[0]
    param = SPDParameter(A.clone())
    target = torch.eye(A.size(-1), dtype=A.dtype)

    loss = F.mse_loss(param, target)
    loss.backward()

    dW = param.grad.clone()
    update_parameter(param, dW, lr=1e-2, spd_metric="airm")

    assert is_spd(param.data), "SPD update did not yield SPD matrix."
