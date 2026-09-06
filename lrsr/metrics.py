import torch


#region Distances

def compute_pairwise_angular_distances(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Computes pairwise subspace angular distance (1 - cos^2(theta)) between columns of A and B."""
    # Compute norms squared along feature dimension
    norm_sq_A = torch.sum(A ** 2, dim=0, keepdim=True).T
    norm_sq_B = torch.sum(B ** 2, dim=0, keepdim=True)

    # Pairwise dot products
    dots = torch.matmul(A.T, B)

    # Compute angular distances
    cos_sq = (dots ** 2) / (norm_sq_A * norm_sq_B + eps)
    angular_dist = 1.0 - cos_sq

    return torch.clamp(angular_dist, min=0.0, max=1.0)


def compute_median_angular_distances(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Computes angular distance (1 - cos^2(theta)) between each column and the median profile."""
    # Compute median profile
    median_profile = torch.median(matrix, dim=1, keepdim=True)[0]

    # Compute pairwise angular distances
    return compute_pairwise_angular_distances(median_profile, matrix, eps=eps).squeeze(0)

#endregion

#region Errors

def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.nn.functional.mse_loss(y_pred, y_true).item()

def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.nn.functional.l1_loss(y_pred, y_true).item()

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(torch.nn.functional.mse_loss(y_pred, y_true)).item()

def mape(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    return torch.mean(torch.abs((y_true - y_pred) / (y_true.abs() + eps))).item()

def wape(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    return (torch.sum(torch.abs(y_true - y_pred)) / (torch.sum(torch.abs(y_true)) + eps)).item()

#endregion
