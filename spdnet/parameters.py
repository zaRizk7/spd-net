from torch import eye, matmul
from torch.linalg import matrix_norm
from torch.nn import Parameter

__all__ = ["SemiOrthogonalParameter", "SPDParameter"]


class SemiOrthogonalParameter(Parameter):
    """
    A parameter constrained to lie on the (batched) Stiefel manifold with orthonormal columns.

    For each [..., n, p] matrix in the batch, we expect:
        - n ≥ p (tall matrix)
        - XᵀX = I_p (column-wise orthonormality)

    This class supports multi-head or batched parameters by checking only the last two dimensions.
    It assumes the Stiefel constraint is handled during training (e.g., via QR retraction, Householder flows).

    Args:
        data (torch.Tensor): Tensor of shape [..., n, p]
        requires_grad (bool): Whether gradients should be tracked.
    """

    def __new__(cls, data, requires_grad=True):
        if data.ndim < 2:
            raise ValueError(f"SemiOrthogonalParameter requires at least 2D input, got shape {data.shape}")
        return super().__new__(cls, data, requires_grad)

    def __repr__(self):
        data = self.data
        *_, n, p = data.shape
        shape = "XᵀX - I_p"
        if n < p:
            data = data.mT  # transpose last two dims
            n, p = p, n
            shape = "XᵀX - I_n"
        I = eye(p, dtype=data.dtype, device=data.device)  # noqa: E741
        WtW = matmul(data.mT, data)
        deviation = matrix_norm(WtW - I, ord="fro").mean().item()
        return (
            f"Parameter containing:\n{data.__repr__()}\n"
            f"Mean frobenius norm of ({shape}): {deviation:.4e}\n"
            f"Shape: {data.shape}"
        )


class SPDParameter(Parameter):
    """
    A parameter constrained to lie on the (batched) SPD manifold.

    For each [..., n, n] matrix in the batch:
        - The matrix should be symmetric: X = Xᵀ
        - Eigenvalues should be strictly positive (not checked here)

    This class supports batched or multi-head SPD parameters.
    SPD constraints must be enforced during training (e.g., via EVD rectification, exponential map).

    Args:
        data (torch.Tensor): Tensor of shape [..., n, n]
        requires_grad (bool): Whether gradients should be tracked.
    """

    def __new__(cls, data, requires_grad=True):
        if data.ndim < 2 or data.shape[-2] != data.shape[-1]:
            raise ValueError(f"SPDParameter must have square trailing dimensions, got shape {data.shape}")
        return super().__new__(cls, data, requires_grad)

    def __repr__(self):
        data = self.data
        deviation = matrix_norm(data - data.mT, ord="fro").mean().item()
        return (
            f"Parameter containing:\n{data.__repr__()}\n"
            f"Frobenius norm of (X - Xᵀ): {deviation:.4e}\n"
            f"Shape: {data.shape}"
        )
