import torch
from torch import Tensor
from torch.optim import Optimizer

from ..parameters import SemiOrthogonalParameter, SPDParameter
from .update_rule import update_parameter

__all__ = ["Adam"]


class Adam(Optimizer):
    r"""
    Implements the Adam optimization algorithm with support for manifold-aware updates.

    This variant of Adam supports three types of parameters:

    - **SemiOrthogonalParameter** (Stiefel manifold): updated using QR-based retraction or landing algorithm.
    - **SPDParameter** (Symmetric Positive Definite manifold): updated via exponential map using a specified Riemannian metric.
    - **Standard parameters**: updated with conventional Adam (optionally with AMSGrad and decoupled weight decay).

    This is not a Riemannian Adam optimizer. Instead, it applies standard Adam updates followed by
    projection or retraction onto the corresponding manifold.

    References:
        - Diederik P. Kingma and Jimmy Ba. 2015. "Adam: A Method for Stochastic Optimization"

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: 1e-3.
        betas (Tuple[float, float], optional): Coefficients used for computing running averages of
            gradient and its square. Default: (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.0.
        decoupled_weight_decay (bool, optional): If True, uses decoupled weight decay (as in AdamW). Default: True.
        orth_update_rule (str, optional): Update rule for semi-orthogonal parameters.
            One of {"retraction", "landing", None}. Default: "retraction".
        landing (float, optional): Scaling coefficient for the landing update (used only if orth_update_rule="landing"). Default: 1.0.
        eps_landing (float, optional): Maximum norm of `norm(X.T @ X - I)` to estimate safe learning rate
            (used only if orth_update_rule="landing"). Default: 0.5.
        spd_metric (str, optional): Riemannian metric for SPD updates.
            One of {"airm", "lem", "euc"}. Default: "airm".
        amsgrad (bool, optional): Whether to use the AMSGrad variant of Adam. Default: False.
        maximize (bool, optional): Maximize the objective function instead of minimizing. Default: False.

    Example::
        >>> optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

    Note:
        - This optimizer assumes special parameters (e.g. `SemiOrthogonalParameter`, `SPDParameter`)
          are explicitly declared using custom parameter types.
        - The actual parameter-specific update is handled via `update_parameter`, which applies
          appropriate projection or retraction for each manifold.
        - While it applies manifold-specific updates, this optimizer is **not** a full Riemannian optimizer.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        decoupled_weight_decay=True,
        orth_update_rule="retraction",
        landing=1.0,
        eps_landing=0.5,
        spd_metric="airm",
        amsgrad=False,
        *,
        maximize=False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor learning rate must be scalar")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta values: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if orth_update_rule not in {"retraction", "landing", None}:
            raise ValueError(f"Invalid orth_update_rule: {orth_update_rule}")
        if landing < 0.0:
            raise ValueError(f"Invalid landing term: {landing}")
        if eps_landing < 0.0:
            raise ValueError(f"Invalid eps_landing value: {eps_landing}")
        if spd_metric not in {"airm", "lem", "euc"}:
            raise ValueError(f"Invalid SPD metric: {spd_metric}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            orth_update_rule=orth_update_rule,
            landing=landing,
            eps_landing=eps_landing,
            spd_metric=spd_metric,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("decoupled_weight_decay", True)
            group.setdefault("orth_update_rule", None)
            group.setdefault("spd_metric", "airm")
            group.setdefault("maximize", False)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decoupled_weight_decay = group["decoupled_weight_decay"]
            orth_update_rule = group["orth_update_rule"]
            landing = group["landing"]
            eps_landing = group["eps_landing"]
            spd_metric = group["spd_metric"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                if maximize:
                    grad *= -1

                is_semi_orthogonal = isinstance(param, SemiOrthogonalParameter)
                is_spd = isinstance(param, SPDParameter)
                is_standard_param = not (is_semi_orthogonal or is_spd)

                if is_standard_param and weight_decay != 0 and not decoupled_weight_decay:
                    grad.add_(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)

                step_size = lr / bias_correction1

                if is_standard_param and weight_decay != 0 and decoupled_weight_decay:
                    grad.add_(param, alpha=weight_decay)

                adapted_grad = exp_avg.div(denom)

                update_parameter(
                    param,
                    adapted_grad,
                    step_size,
                    orth_update_rule,
                    landing,
                    eps_landing,
                    spd_metric,
                )

        return loss
