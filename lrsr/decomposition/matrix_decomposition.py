import torch
from typing import Tuple
from abc import ABC, abstractmethod


class MatrixDecomposition(ABC):
    """Abstract base class for low-rank matrix decomposition algorithms."""

    @abstractmethod
    def decompose(self, matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decomposes an input matrix into factor tensors (U, S, Vh) such that: Matrix ≈ U @ diag(S) @ Vh"""
        pass
