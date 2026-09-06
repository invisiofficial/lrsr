import torch
import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from typing import Callable, Optional

from .data_clusterization import DataClusterization


class KMeansClusterization(DataClusterization):
    """Constrained K-Means clustering using Min-Cost Flow LP."""

    def __init__(
        self,
        distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        tolerance: float = 0.01,
        max_iter: int = 20,
    ):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.distance_fn = distance_fn
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

        # Initialize centroids
        data_np = self.data_.cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        kmeans.fit(data_np.T)
        centroids = torch.tensor(kmeans.cluster_centers_.T, dtype=self.data_.dtype, device=self.data_.device)
        centroids = centroids / (torch.norm(centroids, dim=0, keepdim=True) + 1e-12)

        labels = np.full(num_vectors, -1, dtype=int)

        for _ in range(self.max_iter):
            # E-step: Pairwise distances of shape (num_vectors, num_clusters)
            cost_matrix = self.distance_fn(self.data_, centroids).cpu().numpy()
            c = cost_matrix.flatten().astype(np.float64)
            n_vars = num_vectors * num_clusters

            # Equality constraints: Each vector is assigned to exactly one cluster
            A_eq = np.zeros((num_vectors, n_vars), dtype=np.float64)
            b_eq = np.ones(num_vectors, dtype=np.float64)
            for i in range(num_vectors):
                A_eq[i, i * num_clusters : (i + 1) * num_clusters] = 1.0

            # Inequality constraints: Cluster size bounds [size_min, size_max]
            A_ub = np.zeros((2 * num_clusters, n_vars), dtype=np.float64)
            b_ub = np.zeros(2 * num_clusters, dtype=np.float64)

            for k in range(num_clusters):
                indices = np.arange(k, n_vars, num_clusters)
                A_ub[k, indices] = -1.0
                b_ub[k] = -size_min
                A_ub[num_clusters + k, indices] = 1.0
                b_ub[num_clusters + k] = size_max

            bounds = [(0.0, 1.0)] * n_vars

            # Solve assignment LP via HiGHS solver
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

            if result.success:
                x = result.x.reshape(num_vectors, num_clusters)
                new_labels = np.argmax(np.round(x, decimals=5), axis=1)
            else:
                new_labels = np.argmin(cost_matrix, axis=1)

            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # M-step: Update and re-normalize centroids
            for c_idx in range(num_clusters):
                mask = torch.from_numpy(labels == c_idx).to(self.data_.device)
                cluster_points = self.data_[:, mask]
                if cluster_points.shape[1] > 0:
                    new_c = torch.mean(cluster_points, dim=1)
                    centroids[:, c_idx] = new_c / (torch.norm(new_c) + 1e-12)

        return labels
