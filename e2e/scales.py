import torch
from typing import Tuple

from lrsr.utils import compute_otsu_threshold
from lrsr.metrics import compute_pairwise_angular_distances, compute_median_angular_distances
from lrsr.decomposition.alternating_decomposition import AlternatingDecomposition
from lrsr.approximation.rank_approximation import RankApproximation
from lrsr.clusterization.kmeans_clusterization import KMeansClusterization

# Global approximator
rank_approximator = RankApproximation(AlternatingDecomposition(num_iterations=3))

#region Helpers

def lowrank_factors(S: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns u, v such that outer(u, v) is the low-rank scale matrix."""
    U, singular_values, Vh = rank_approximator.decomposition_method.decompose(S, rank=1)
    u = U[:, 0] * singular_values[0]
    v = Vh[0]
    return torch.clamp(u, min=eps), torch.clamp(v, min=eps)

#endregion

def scales_per_channel(W: torch.Tensor, qmax: float, eps: float = 1e-7) -> torch.Tensor:
    """Returns one scale for every output feature."""
    return torch.clamp(W.abs().max(dim=0)[0] / qmax, min=eps)

def scales_per_group(W: torch.Tensor, group_size: int, qmax: float, eps: float = 1e-7) -> torch.Tensor:
    """Returns a `(input_groups, out_features)` scale matrix with one entry per input group."""
    n, p = W.shape
    if n % group_size != 0:
        raise ValueError(f"in_features ({n}) must be divisible by group_size ({group_size}).")
    groups = W.view(n // group_size, group_size, p)
    return torch.clamp(groups.abs().max(dim=1)[0] / qmax, min=eps)

def scales_lrsr_naive(S: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns u, v such that outer(u, v) is the low-rank approximation of S."""
    return lowrank_factors(S, eps)

def scales_lrsr_1dos(S: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns u_main, u_minor, v and the minor-feature mask from an threshold split."""
    dist = compute_median_angular_distances(S)
    tau = compute_otsu_threshold(dist.cpu().numpy())
    minor_mask = torch.as_tensor(dist.cpu().numpy() > tau, device=S.device)
    main_mask = ~minor_mask

    u_main = torch.ones(S.size(0), device=S.device, dtype=S.dtype)
    v = torch.ones(S.size(1), device=S.device, dtype=S.dtype)
    if main_mask.any():
        u_main, v_main = lowrank_factors(S[:, main_mask], eps)
        v[main_mask] = v_main

    u_minor = u_main
    if minor_mask.any():
        u_minor, v_minor = lowrank_factors(S[:, minor_mask], eps)
        v[minor_mask] = v_minor

    return u_main, u_minor, v, minor_mask

def scales_lrsr_kmeans(S: torch.Tensor, clusters: int, tolerance: float, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns per-cluster input scales, per-feature output scales and feature cluster labels."""
    model = KMeansClusterization(tolerance=tolerance, max_iter=10, n_init=3)
    model.fit(S)
    labels = torch.as_tensor(model.predict(clusters), device=S.device, dtype=torch.long)
    input_scales = torch.ones(clusters, S.size(0), device=S.device, dtype=S.dtype)
    output_scales = torch.ones(S.size(1), device=S.device, dtype=S.dtype)

    for cluster in range(clusters):
        mask = labels == cluster
        if mask.any():
            input_scale, cluster_output_scale = lowrank_factors(S[:, mask], eps)
            input_scales[cluster] = input_scale
            output_scales[mask] = cluster_output_scale

    return input_scales, output_scales, labels
