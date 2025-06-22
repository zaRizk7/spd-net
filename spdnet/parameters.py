from torch.nn import Parameter

__all__ = ["SemiOrthogonalParameter", "SPDParameter"]


class SemiOrthogonalParameter(Parameter):
    """
    A parameter constrained to lie on the Stiefel manifold, specifically with orthonormal columns.

    The Stiefel manifold St(n, p) is the set of all n × p matrices whose columns are orthonormal:
        St(n, p) = { X ∈ ℝ^{n×p} | XᵀX = I_p }

    In this implementation, we assume a "semi-orthogonal" setting, where:
        - The input tensor is 2D (shape [n, p])
        - n ≥ p (tall matrix)
        - Only column-wise orthogonality is required (XᵀX = I)

    This is useful in models where preserving orthonormality helps improve conditioning,
    gradient flow, or satisfies geometric constraints, such as:
        - Variational autoencoders
        - SPDNet and manifold learning
        - Riemannian optimization layers

    This class only validates the dimensionality during initialization.
    The actual orthonormality must be enforced during training (e.g., via QR retraction or Householder parametrization).
    Alternatively, trivialization methods can also be applied without using this class.

    Args:
        data (torch.Tensor): Initial 2D tensor.
        requires_grad (bool): Whether gradients should be tracked.
    """

    def __new__(cls, data, requires_grad=True):
        if data.ndim != 2:
            raise ValueError(f"SemiOrthogonalParameter only supports 2D tensors, got shape {data.shape}")
        return super().__new__(cls, data, requires_grad)

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()


class SPDParameter(Parameter):
    """
    A parameter constrained to lie in the space of Symmetric Positive Definite (SPD) matrices.

    The SPD manifold consists of symmetric matrices with strictly positive eigenvalues:
        SPD(n) = { X ∈ ℝ^{n×n} | X = Xᵀ, and all eigenvalues(λ_i) > 0 }

    SPD matrices arise in:
        - Covariance and kernel matrices
        - Riemannian geometry (e.g., Log-Euclidean metric)
        - Deep learning layers using second-order statistics

    This class validates:
        - Input is a square 2D tensor (n × n)
        - Positive definite and symmetry isn't checked here since it accepts `torch.empty` tensors.

    Note:
        The SPD condition is checked only at initialization.
        During training, you are responsible for maintaining SPD-ness
        using proper update strategies (e.g., EVD clamping, matrix exponential, or affine-invariant retractions).

    Args:
        data (torch.Tensor): Initial SPD matrix (must be symmetric and PD).
        requires_grad (bool): Whether gradients should be tracked.
    """

    def __new__(cls, data, requires_grad=True):
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f"SPDParameter must be a square 2D matrix, got shape {data.shape}")

        return super().__new__(cls, data, requires_grad)

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()
