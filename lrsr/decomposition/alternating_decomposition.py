from typing import Tuple
import torch

from .matrix_decomposition import MatrixDecomposition


class AlternatingDecomposition(MatrixDecomposition):
    """Computes a strictly upper-bounding Rank-1 decomposition (u * v^T >= S) using alternating max-scale updates."""

    def __init__(self, num_iterations: int = 20):
        self.num_iterations = num_iterations

    def decompose(self, matrix: torch.Tensor, rank: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decomposes matrix into U, S, Vh such that outer(U, Vh) >= matrix."""
        if rank != 1:
            print("Warning: AlternatingDecomposition is strictly formulated for Rank-1. Forcing rank=1.")

        S_mat = torch.clamp(matrix, min=1e-7)
        rows, _ = S_mat.shape
        device = S_mat.device
        dtype = S_mat.dtype

        # Initialize u with ones
        u = torch.ones(rows, device=device, dtype=dtype)

        # Execute Alternating Optimization
        for _ in range(self.num_iterations):
            v = torch.max(S_mat / u.unsqueeze(1), dim=0)[0]

            u = torch.max(S_mat / v.unsqueeze(0), dim=1)[0]

        # Reshape factors into standard format
        U = u.unsqueeze(1)
        S = torch.ones(1, device=device, dtype=dtype)
        Vh = v.unsqueeze(0)

        return U, S, Vh
