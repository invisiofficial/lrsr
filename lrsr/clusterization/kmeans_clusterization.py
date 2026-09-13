import torch
import numpy as np
from typing import Optional
from k_means_constrained import KMeansConstrained

from .data_clusterization import DataClusterization


class KMeansClusterization(DataClusterization):
    """Constrained K-Means clustering."""

    def __init__(
        self,
        tolerance: float = 0.01,
        max_iter: int = 20,
        n_init: int = 5,
    ):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.n_init = n_init
        self.data_: Optional[torch.Tensor] = None

    def fit(self, data: torch.Tensor):
        """Stores the scale matrix for clustering."""
        self.data_ = data
        return self

    def predict(self, num_clusters: int) -> np.ndarray:
        """Partitions columns into capacity-bounded clusters."""
        if self.data_ is None:
            raise ValueError("The model has not been fitted yet. Call fit() first.")

        _, num_vectors = self.data_.shape

        # Compute cluster capacity bounds
        target_size = num_vectors / num_clusters
        size_min = int(np.floor(target_size * (1.0 - self.tolerance)))
        size_max = int(np.ceil(target_size * (1.0 + self.tolerance)))

        if size_min * num_clusters > num_vectors:
            size_min = max(1, num_vectors // num_clusters - 1)
        if size_max * num_clusters < num_vectors:
            size_max = num_vectors // num_clusters + 1

        data_np = self.data_.detach().cpu().numpy().T

        # Compute L2 normalization
        norms = np.linalg.norm(data_np, axis=1, keepdims=True)
        data_np = np.divide(data_np, norms, out=np.zeros_like(data_np), where=norms!=0)

        # Initialize clusterization
        kmeans = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=size_min,
            size_max=size_max,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=0,
        )

        # Compute clusters
        labels = kmeans.fit_predict(data_np)
        return labels
