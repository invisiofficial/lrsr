import numpy as np
from abc import ABC, abstractmethod


class DataClusterization(ABC):
    """Abstract base class for data-driven clusterization methods."""

    @abstractmethod
    def fit(self, distance_matrix: np.ndarray):
        """Builds the linkage tree from a pairwise distance matrix."""
        pass

    @abstractmethod
    def predict(self, num_clusters: int) -> np.ndarray:
        """Cuts the precomputed tree at the given number of clusters."""
        pass
