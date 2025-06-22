import torch
from torch.linalg import matrix_norm

from ..metrics import exp_map
from ..parameters import SemiOrthogonalParameter, SPDParameter
from ..functions import skew

__all__ = ["update_parameter"]


def update_parameter(param, grad, lr, orth_update_rule=None, landing=1.0, eps=0.5, spd_metric="airm"):
    r"""
    Applies an in-place update to a parameter based on its type (Euclidean, Semi-Orthogonal, or SPD).

    Args:
        param (torch.nn.Parameter): Parameter tensor to update.
        grad (torch.Tensor): Gradient tensor.
        lr (float): Learning rate.
        orth_update_rule (str, optional): Update rule for semi-orthogonal parameters.
            One of {"retraction", "landing", None}.
        landing (float, optional): Landing coefficient (only used if `orth_update_rule="landing"`).
        eps (float, optional): Maximum norm of `norm(X.T @ X - I)` to estimate safe learning rate (only used if `orth_update_rule="landing"`).
        spd_metric (str, optional): Riemannian metric for SPD update ("airm", "lem", or "euc").
    """
    if isinstance(param, SemiOrthogonalParameter):
        if orth_update_rule == "retraction":
            _update_retraction_semi_orthogonal_parameters(param, grad, lr)
        elif orth_update_rule == "landing":
            _update_landing_semi_orthogonal_parameters(param, grad, lr, landing, eps)
        else:
            param.data.add_(grad, alpha=-lr)
    elif isinstance(param, SPDParameter):
        _update_spd_parameters(param, grad, lr, spd_metric)
    else:
        param.data.add_(grad, alpha=-lr)


def _update_retraction_semi_orthogonal_parameters(param, grad, lr):
    r"""
    In-place retraction update for parameters on the Stiefel manifold.

    This update projects the Euclidean gradient onto the tangent space of the manifold:
        grad ← grad - grad @ (XᵀX)

    Then, it performs a gradient descent step followed by a QR-based retraction
    to ensure the updated matrix remains on the Stiefel manifold.

    Args:
        param (SemiOrthogonalParameter): Weight constrained to the Stiefel manifold.
        grad (torch.Tensor): Euclidean gradient of the loss w.r.t. `param`.
        lr (float): Learning rate.

    References:
        - Absil, P.-A., Mahony, R., & Sepulchre, R. (2008).
          *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.

        - Huang, Zhiwu, and Luc Van Gool. (2017).
          *A Riemannian Network for SPD Matrix Learning*. Proceedings of AAAI 2017.
          https://arxiv.org/pdf/1608.04233
    """
    n, p = param.shape[-2:]
    param_data = param.data
    grad = grad.data

    # QR will truncate the larger dimension if n < p, so we transpose the data
    # and gradient tensors before the update and transpose them back afterwards
    # to preserve the correct shape for the Stiefel manifold.
    if n < p:
        param_data = param_data.mT
        grad = grad.mT

    # Project gradient onto the tangent space of the manifold
    # dX <- dX - dX @ (XᵀX)
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


def _update_landing_semi_orthogonal_parameters(param, grad, lr, landing=1.0, eps=0.5):
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
        eps (float): Maximum norm of norm(X.T @ X - I) to estimate safe learning rate.

    Reference:
        Ablin, P., & Peyré, G. (2022). *Landing: Directly Optimizing Embeddings on Manifolds*.
        In Proceedings of the 39th International Conference on Machine Learning (ICML).
        https://proceedings.mlr.press/v151/ablin22a/ablin22a.pdf
    """
    # Cast lr and eps to the same dtype and device as param to use torch.minimum/maximum
    lr = torch.as_tensor(lr, dtype=param.dtype, device=param.device)
    eps = torch.as_tensor(eps, dtype=param.dtype, device=param.device)

    n, p = param.shape[-2:]
    param_data = param.data
    grad = grad.data

    if n > p:
        param_data = param_data.mT
        grad = grad.mT

    # Ψ(X) = SkewSymmetric(dX @ Xᵀ) = 0.5 * (dX @ Xᵀ - X @ dXᵀ)
    psi = torch.matmul(grad, param_data.mT)
    psi = skew(psi)
    # Λ_proj = Ψ(X) @ X
    torch.matmul(psi, param_data, out=grad)  # reuse grad for Λ

    identity = torch.eye(min(n, p), dtype=param.dtype, device=param.device)
    gram = torch.matmul(param_data, param_data.mT)

    # Correction: ∇N(X)=(X Xᵀ - I) @ X
    correction = gram - identity
    correction = torch.matmul(correction, param_data)

    # Λ(X) = Ψ(X) @ X + λ * ∇N(X)
    grad.add_(correction, alpha=landing)

    # Compute safe step size (Preposition 6 in Ablin & Peyré, 2022)
    # 1. Compute the frobenius norm d=N(X)=0.25*||X Xᵀ - I||_F^2
    d = matrix_norm(gram - identity) ** 2 / 4
    # 2. Compute a=||Ψ(X)||_F
    a = matrix_norm(psi)
    # 3. Compute α=2 * λ * d - 2 * a * d - 2 * λ * d^2
    alpha = 2 * landing * d - 2 * a * d - 2 * landing * d**2
    # 4. Compute β=a^2 + λ^2d^3 + 2λad^2 + a^2d
    beta = a**2 + landing**2 * d**3 + 2 * landing * a * d**2 + a**2 * d
    # 5. Compute η∗(a, d)=(sqrt(α^2 + 4β(ε−d)) + α) / (2β)
    # Remember that it is still an multi-dimension tensor when param.ndim > 2
    safe_lr = (torch.sqrt(alpha**2 + 4 * beta * (eps - d)) + alpha) / (2 * beta)
    # 6. Apply min(η∗(a, d), η) to ensure the step size is not too large
    safe_lr = torch.minimum(safe_lr, lr)

    if n > p:
        grad = grad.mT
        if safe_lr.ndim > 1:
            # keep the shape for broadcasting (..., 1)
            # example if we have (n_head, n, p) param size
            # safe_lr will be (n_head), as n and p are gone
            # due to matrix_norm
            # to make it broadcastable, we need to add a new dimension
            safe_lr = safe_lr[..., None]

    # Update parameter: X_{k+1} ← X_k - η_k * Λ(X_k)
    param.data.add_(grad, alpha=-safe_lr)


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

    # Riemannian tangential projection operator: G ← P @ 0.5 * (G + G.T) @ P
    grad = (grad + grad.mT) / 2
    torch.matmul(param_data, grad, out=grad)
    torch.matmul(grad, param_data, out=grad)

    # Exponential map update
    param.data.copy_(exp_map(grad.mul_(-lr), param_data, metric))
