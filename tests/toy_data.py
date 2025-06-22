import numbers
from typing import Iterable

import torch


def make_spd_matrix(size, device=None, dtype=None):
    r"""
    Generates a random symmetric positive definite (SPD) matrix of shape `(size, size)`.

    This function constructs an SPD matrix by:
        1. Generating a random matrix `A`.
        2. Forming a symmetric matrix `A.T @ A`.
        3. Applying SVD to orthogonalize the result and stabilize its spectrum.
        4. Adding a diagonal matrix with values > 1 to ensure positive definiteness.

    Args:
        size (int): Size of the square matrix (n × n).
        device (torch.device, optional): Device on which to allocate the tensor.
        dtype (torch.dtype, optional): Desired floating point type of returned tensor.

    Returns:
        torch.Tensor: A symmetric positive definite matrix of shape `(size, size)`.
    """
    # Create a random matrix and use SVD to ensure it is positive definite
    factory_kwargs = {"device": device, "dtype": dtype}
    A = torch.rand(size, size, **factory_kwargs)
    U, _, Vt = torch.linalg.svd(A.T @ A)
    A = U @ (1 + torch.diag(torch.rand(size, **factory_kwargs))) @ Vt

    return A


def make_blobs(
    n_samples=100,
    n_features=2,
    centers=None,
    cluster_std=1.0,
    center_box=(-10.0, 10.0),
    shuffle=True,
    return_centers=False,
    device="cpu",
):
    r"""
    Generate isotropic Gaussian blobs for clustering using PyTorch distributions.

    Args:
        n_samples (int or Iterable[int], optional):
            If int, total number of samples to generate, equally divided among centers.
            If list, number of samples per center. Default: 100.

        n_features (int, optional):
            Number of features (dimensions) for each sample. Default: 2.

        centers (int or Tensor of shape (n_centers, n_features), optional):
            Number of cluster centers or their fixed locations. If None, 3 centers are
            generated randomly in `center_box`. Default: None.

        cluster_std (float or Iterable[float], optional):
            Standard deviation of each cluster. If a float, all clusters have the same std.
            If a list or tensor, must match the number of centers. Default: 1.0.

        center_box (Tuple[float, float], optional):
            Bounding box (min, max) for random center initialization if `centers` is int or None.
            Default: (-10.0, 10.0).

        shuffle (bool, optional):
            Whether to shuffle the samples after generation. Default: True.

        return_centers (bool, optional):
            Whether to return the actual center coordinates. Default: False.

        device (str or torch.device, optional):
            Device on which tensors will be created. Default: "cpu".

    Returns:
        Tuple:
            - X (Tensor): Samples of shape (sum(n_samples), n_features)
            - y (Tensor): Cluster labels of shape (sum(n_samples),)
            - centers (Tensor): Cluster centers of shape (n_centers, n_features),
              only if `return_centers=True`
    """
    if isinstance(n_samples, numbers.Integral):
        if centers is None:
            n_centers = 3
        elif isinstance(centers, numbers.Integral):
            n_centers = centers
        else:
            centers = torch.as_tensor(centers, dtype=torch.float32, device=device)
            n_centers = centers.size(0)
            n_features = centers.size(1)

        if centers is None or isinstance(centers, int):
            centers = torch.empty(n_centers, n_features, device=device).uniform_(*center_box)
    else:
        n_centers = len(n_samples)
        if centers is None:
            centers = torch.empty(n_centers, n_features, device=device).uniform_(*center_box)
        else:
            centers = torch.as_tensor(centers, dtype=torch.float32, device=device)
            if centers.size(0) != n_centers:
                raise ValueError(f"Length mismatch: centers={centers.size(0)} vs n_samples={n_centers}")
            n_features = centers.size(1)

    if isinstance(cluster_std, numbers.Real):
        cluster_std = torch.full((n_centers,), cluster_std, dtype=torch.float32, device=device)
    else:
        cluster_std = torch.as_tensor(cluster_std, dtype=torch.float32, device=device)
        if cluster_std.size(0) != n_centers:
            raise ValueError("cluster_std must match number of centers")

    if isinstance(n_samples, Iterable):
        n_samples_per_center = list(n_samples)
    else:
        base = n_samples // n_centers
        remainder = n_samples % n_centers
        n_samples_per_center = [base + (1 if i < remainder else 0) for i in range(n_centers)]

    total_samples = sum(n_samples_per_center)
    X = torch.empty((total_samples, n_features), dtype=torch.float32, device=device)
    y = torch.empty(total_samples, dtype=torch.long, device=device)

    start = 0
    for i, (mean, std, n) in enumerate(zip(centers, cluster_std, n_samples_per_center)):
        dist = torch.distributions.Normal(loc=mean, scale=std)
        X[start : start + n] = dist.sample((n,))
        y[start : start + n] = i
        start += n

    if shuffle:
        idx = torch.randperm(total_samples, device=device)
        X = X[idx]
        y = y[idx]

    if return_centers:
        return X, y, centers
    return X, y
