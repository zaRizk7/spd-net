import torch
from torch import Tensor
from torch.optim import Optimizer

from ..parameters import SemiOrthogonalParameter, SPDParameter
from .update_rule import update_parameter

__all__ = ["SGD"]


class SGD(Optimizer):
    r"""
    Implements stochastic gradient descent (optionally with momentum) and supports parameters
    constrained on specific manifolds such as:

    - **Stiefel manifold** (`SemiOrthogonalParameter`): uses QR retraction or landing update
    - **SPD manifold** (`SPDParameter`): uses exponential map update
    - **Euclidean space**: standard SGD with optional momentum and weight decay

    This is not a Riemannian SGD optimizer. Instead, it applies standard SGD updates followed by
    projection or retraction onto the corresponding manifold.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate (default: 1e-3).
        momentum (float, optional): momentum factor (default: 0).
        dampening (float, optional): dampening for momentum (default: 0).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        nesterov (bool, optional): enables Nesterov momentum (default: False).
        decoupled_weight_decay (bool, optional): if True, applies weight decay as a separate step (default: True).
        orth_update_rule (str, optional): Update rule for semi-orthogonal parameters.
            One of {"retraction", "landing", None}. Default: "retraction".
        landing (float, optional): Scaling coefficient for the landing update (used only if `orth_update_rule="landing"`, default: 1.0).
        eps (float, optional): Maximum norm of `norm(X.T @ X - I)` to estimate safe learning rate
            (used only if `orth_update_rule="landing"`, default: 0.5).
        spd_metric (str, optional): Riemannian metric for SPD updates.
            One of {"airm", "lem", "euc"} (default: "airm").
        maximize (bool, optional): maximize the objective instead of minimizing (default: False).

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> optimizer.step()
        >>> optimizer.zero_grad()

    Note:
        - Momentum and Nesterov updates only apply to Euclidean parameters.
        - If `orth_update_rule` is set and a parameter is a `SemiOrthogonalParameter`,
          its gradient will be projected accordingly.
        - `SPDParameter` updates will follow exponential map based on the given `spd_metric`.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        decoupled_weight_decay=True,
        orth_update_rule="retraction",
        landing=1.0,
        eps=0.5,
        spd_metric="airm",
        *,
        maximize=False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor learning rate must be scalar")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")
        if orth_update_rule not in {"retraction", "landing", None}:
            raise ValueError(f"Invalid orth_update_rule: {orth_update_rule}")
        if landing < 0.0:
            raise ValueError(f"Invalid landing term: {landing}")
        if spd_metric not in {"airm", "lem", "euc"}:
            raise ValueError(f"Invalid SPD metric: {spd_metric}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            decoupled_weight_decay=decoupled_weight_decay,
            orth_update_rule=orth_update_rule,
            landing=landing,
            eps=eps,
            spd_metric=spd_metric,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
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
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            decoupled_weight_decay = group["decoupled_weight_decay"]
            orth_update_rule = group["orth_update_rule"]
            landing = group["landing"]
            eps = group["eps"]
            spd_metric = group["spd_metric"]
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

                # Apply standard weight decay (L2 penalty)
                if is_standard_param and weight_decay != 0 and not decoupled_weight_decay:
                    grad = grad.add(param, alpha=weight_decay)

                # Apply momentum
                if momentum != 0:
                    state = self.state[param]
                    buf = state.get("momentum_buffer", None)
                    if buf is None:
                        buf = torch.clone(grad).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                # Decoupled weight decay (à la AdamW)
                if is_standard_param and weight_decay != 0 and decoupled_weight_decay:
                    grad = grad.add(param, alpha=weight_decay)

                # Dispatch to update rule
                update_parameter(param, grad, lr, orth_update_rule, landing, eps, spd_metric)

        return loss
