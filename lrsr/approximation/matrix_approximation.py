import torch
from abc import ABC, abstractmethod


class MatrixApproximation(ABC):
    """Abstract base class for matrix approximation estimators."""

    @abstractmethod
    def approximate(self, matrix: torch.Tensor, rank: int) -> torch.Tensor:
        """Approximates the given matrix with a specified rank."""
        pass
