import torch

from .matrix_approximation import MatrixApproximation
from ..decomposition.matrix_decomposition import MatrixDecomposition


class RankApproximation(MatrixApproximation):
    """Reconstructs a low-rank matrix approximation by leveraging a given matrix decomposition strategy."""

    def __init__(self, decomposition_method: MatrixDecomposition):
        self.decomposition_method = decomposition_method

    def approximate(self, matrix: torch.Tensor, rank: int) -> torch.Tensor:
        """Computes decomposition and reconstructs matrix: U @ diag(S) @ Vh."""
        # Decompose original matrix
        U, S, Vh = self.decomposition_method.decompose(matrix, rank)

        # Reconstruct approximation
        S_diag = torch.diag(S)
        approx_matrix = torch.matmul(U, torch.matmul(S_diag, Vh))

        return approx_matrix
