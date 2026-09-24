from typing import Tuple
import torch

from .matrix_decomposition import MatrixDecomposition


class UpperboundingDecomposition(MatrixDecomposition):
    """Computes a strictly upper-bounding Rank-1 decomposition."""

    def decompose(self, matrix: torch.Tensor, rank: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decomposes matrix into U, S, Vh such that outer(Vh, U) >= matrix."""
        if rank != 1:
            print("Warning: UpperboundingDecomposition is strictly formulated for Rank-1. Forcing rank=1.")

        S_mat = torch.clamp(matrix, min=1e-7)
        device = S_mat.device
        dtype = S_mat.dtype

        # Compute upper-bounding decomposition
        u = torch.max(S_mat, dim=0)[0]
        v = torch.max(S_mat / u.unsqueeze(0), dim=1)[0]

        # Reshape factors into standard format
        U = u.unsqueeze(1)
        S = torch.ones(1, device=device, dtype=dtype)
        Vh = v.unsqueeze(0)

        return U, S, Vh
