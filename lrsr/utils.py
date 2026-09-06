import numpy as np


def compute_otsu_threshold(distances: np.ndarray, bins: int = 128) -> float:
    """Computes the optimal separation threshold using 1D Otsu's Method."""
    # Create histogram
    hist, bin_edges = np.histogram(distances, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Cumulative sums and means
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    weight1[weight1 == 0] = 1e-10
    weight2[weight2 == 0] = 1e-10

    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Compute Between-Class Variance for all possible thresholds
    variance_between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Find the bin center corresponding to the maximum variance
    optimal_idx = np.argmax(variance_between)
    optimal_threshold = bin_centers[optimal_idx]

    return optimal_threshold
