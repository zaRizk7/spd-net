import torch

from ..metrics import exp_map
from ..parameters import SemiOrthogonalParameter, SPDParameter

__all__ = ["update_parameter"]


def update_parameter(param, grad, lr, orth_update_rule=None, landing=1.0, spd_metric="airm"):
    r"""
    Applies an in-place update to a parameter based on its type (Euclidean, Semi-Orthogonal, or SPD).

    Args:
        param (torch.nn.Parameter): Parameter tensor to update.
        grad (torch.Tensor): Gradient tensor.
        lr (float): Learning rate.
        orth_update_rule (str, optional): Update rule for semi-orthogonal parameters.
            One of {"retraction", "landing", None}.
        landing (float, optional): Landing coefficient (only used if `orth_update_rule="landing"`).
        spd_metric (str, optional): Riemannian metric for SPD update ("airm", "lem", or "euc").
    """
    if isinstance(param, SemiOrthogonalParameter):
        if orth_update_rule == "retraction":
            _update_retraction_semi_orthogonal_parameters(param, grad, lr)
        elif orth_update_rule == "landing":
            _update_landing_semi_orthogonal_parameters(param, grad, lr, landing)
        else:
            param.data.add_(grad, alpha=-lr)
    elif isinstance(param, SPDParameter):
        _update_spd_parameters(param, grad, lr, spd_metric)
    else:
        param.data.add_(grad, alpha=-lr)


def _update_retraction_semi_orthogonal_parameters(param, grad, lr):
    r"""
    In-place retraction update for parameters on the Stiefel manifold.

    This update projects the Euclidean gradient onto the tangent space:
        grad ← grad - grad @ (XᵀX)
    Then performs a gradient descent step followed by a QR-based retraction
    to ensure the updated matrix remains on the manifold.

    Args:
        param (SemiOrthogonalParameter): Weight constrained to the Stiefel manifold.
        grad (torch.Tensor): Euclidean gradient of the loss w.r.t. `param`.
        lr (float): Learning rate.

    Reference:
        Huang, Zhiwu, and Luc Van Gool. (2017).
        *A Riemannian Network for SPD Matrix Learning*. Proceedings of AAAI 2017.
        doi:10.1609/aaai.v31i1.10866
        https://arxiv.org/pdf/1608.04233
    """
    n, p = param.shape[-2:]
    param_data = param.data
    grad = grad.data

    if n < p:
        param_data = param_data.mT
        grad = grad.mT

    # Project gradient onto the tangent space of the manifold
    gram = torch.matmul(param_data.mT, param_data)
    correction = torch.matmul(grad, gram)
    grad.sub_(correction)

    # Descent step: tmp ← X - lr * grad (reusing grad as tmp)
    param_data.add_(grad, alpha=-lr)

    # QR retraction back onto the Stiefel manifold
    Q, _ = torch.linalg.qr(param_data)
    if n < p:
        Q = Q.mT

    param.data.copy_(Q)


def _update_landing_semi_orthogonal_parameters(param, grad, lr, landing):
    r"""
    In-place landing update for parameters constrained to the Stiefel manifold.

    The landing algorithm directly performs descent on the Stiefel manifold without requiring
    explicit retraction. It consists of:

    1. A projected gradient via the skew-symmetric component:
        Ψ(X) = 0.5 * (dX @ Xᵀ - X @ dXᵀ)
        Λ_proj = Ψ(X) @ X

    2. A correction term to penalize deviations from orthogonality:
        Λ_corr = (X Xᵀ - I) @ X

    The full update is:
        X ← X - lr * (Λ_proj + λ * Λ_corr)

    Args:
        param (SemiOrthogonalParameter): A parameter constrained to the Stiefel manifold.
        grad (torch.Tensor): Euclidean gradient of the loss w.r.t. `param`.
        lr (float): Learning rate.
        landing (float): Correction strength (λ) enforcing orthogonality.

    Reference:
        Ablin, P., & Peyré, G. (2022). *Landing: Directly Optimizing Embeddings on Manifolds*.
        In Proceedings of the 39th International Conference on Machine Learning (ICML).
        https://proceedings.mlr.press/v151/ablin22a/ablin22a.pdf
    """
    n, p = param.shape[-2:]
    param_data = param.data
    grad = grad.data

    if n < p:
        param_data = param_data.mT
        grad = grad.mT

    # Psi(X) = 0.5 * (dX @ Xᵀ - X @ dXᵀ)
    psi = torch.matmul(grad, param_data.mT)
    psi = (psi - psi.mT) / 2
    torch.matmul(psi, param_data, out=grad)  # reuse grad for Λ

    # Correction: λ * (X Xᵀ - I) X
    eye = torch.eye(param_data.size(0), dtype=param.dtype, device=param.device)
    correction = torch.matmul(param_data, param_data.mT)
    correction.sub_(eye)
    correction = torch.matmul(correction, param_data)

    grad.add_(correction, alpha=landing)

    if n < p:
        grad = grad.mT

    param.data.add_(grad, alpha=-lr)


def _update_spd_parameters(param, grad, lr, metric):
    r"""
    In-place SPD update using the exponential map with affine-invariant transport.

    This ensures the parameter remains on the Symmetric Positive Definite (SPD) manifold by
    applying a Riemannian gradient update of the form:

        param ← Exp_P(-lr * P @ G @ P)

    where:
        - `P` is the current SPD matrix,
        - `G` is the symmetrized Euclidean gradient,
        - `Exp_P` is the exponential map at `P` under the chosen Riemannian metric.

    This approach is described in:

        Brooks et al. (2019). "Riemannian Batch Normalization for SPD Neural Networks."
        In Advances in Neural Information Processing Systems (NeurIPS 2019).
        https://proceedings.neurips.cc/paper_files/paper/2019/file/6e69ebbfad976d4637bb4b39de261bf7-Paper.pdf

    Args:
        param (SPDParameter): Parameter constrained to the SPD manifold.
        grad (torch.Tensor): Euclidean gradient of the loss with respect to `param`.
        lr (float): Learning rate.
        metric (str): Riemannian metric for the exponential map. Options: {"airm", "lem", "euc"}.
    """
    param_data = param.data
    grad = grad.data

    # Ensure symmetry of the gradient
    grad = (grad + grad.mT) / 2

    # Riemannian transport: G ← P @ G @ P
    torch.matmul(param_data, grad, out=grad)
    torch.matmul(grad, param_data, out=grad)

    # Exponential map update
    param.data.copy_(exp_map(grad.mul_(-lr), param_data, metric))
