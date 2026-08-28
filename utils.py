import os
import random
from typing import List, Optional, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import math


CURRENT_SEED = 0

def show_tensor_image(image_tensor: torch.Tensor, label: Optional[str] = None):
    """
    Visualizes a PyTorch tensor as an image.
    """

    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    
    if img.shape[2] == 1:
        img = img.squeeze(2)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    
    if label:
        plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()



def show_tensor_images(
    images: Union[torch.Tensor, List[torch.Tensor]], 
    labels: Optional[Union[str, List[str]]] = None, 
    cols: int = 1,
    base_size: int = 4,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
):
    """
    Visualizes tensors, with optional un-normalization for human-readable viewing.
    """
    if isinstance(images, torch.Tensor) and images.ndim == 3:
        images = [images]
    
    if isinstance(labels, str):
        labels = [labels]
    
    num_images = len(images)
    rows = math.ceil(num_images / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * base_size, rows * base_size))
    axes = np.array(axes).flatten()

    for i in range(num_images):
        img_tensor = images[i].detach().cpu()
        
        img = img_tensor.permute(1, 2, 0).numpy()
        
        if mean is not None and std is not None:
            m = np.array(mean).reshape(1, 1, -1)
            s = np.array(std).reshape(1, 1, -1)
            img = (img * s) + m
            img = np.clip(img, 0, 1)

        if img.shape[2] == 1:
            img = img.squeeze(2)
        
        axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i].axis('off')
        
        if labels and i < len(labels):
            axes[i].set_title(f"{labels[i]}")

    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    CURRENT_SEED = seed


def compute_wasserstein_distance(
    clean_features: torch.Tensor,
    aug_features: torch.Tensor,
    epsilon: float = 0.1,
    max_iterations: int = 100
) -> torch.Tensor:
    """
    Compute Wasserstein distance (Earth Mover's Distance) between clean and augmented
    feature distributions using Sinkhorn iteration for entropic regularized optimal transport.
    
    Args:
        clean_features: Tensor of shape (N, D) - features from clean images
        aug_features: Tensor of shape (M, D) - features from augmented images
        epsilon: Entropic regularization parameter (smaller = closer to exact EMD)
        max_iterations: Maximum Sinkhorn iterations
        
    Returns:
        Tensor of shape (N, M) where entry [i,j] is the Wasserstein distance from
        clean sample i to augmented sample j
    """
    # Compute pairwise Euclidean distance matrix
    # clean_features: (N, D), aug_features: (M, D)
    # dist_matrix: (N, M)
    clean_features = clean_features.contiguous()
    aug_features = aug_features.contiguous()
    
    dist_matrix = torch.cdist(clean_features, aug_features, p=2)
    
    # Sinkhorn iteration for entropic regularized optimal transport
    # Both distributions are uniform (1/N and 1/M)
    N, M = dist_matrix.shape
    
    # Log-domain Sinkhorn for numerical stability
    log_K = -dist_matrix / epsilon  # (N, M)
    
    # Initialize
    log_u = torch.zeros(N, device=dist_matrix.device)
    log_v = torch.zeros(M, device=dist_matrix.device)
    
    # Sinkhorn iterations
    for _ in range(max_iterations):
        # log_u = -logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        # log_v = -logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
        log_v = -torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)
    
    # Compute optimal transport plan
    # P = exp((log_u + log_v - dist_matrix) / epsilon)
    log_plan = log_u.unsqueeze(1) + log_v.unsqueeze(0) - dist_matrix / epsilon
    transport_plan = torch.exp(log_plan)
    
    # Wasserstein distance = sum(P * dist_matrix)
    wasserstein_dist = (transport_plan * dist_matrix).sum(dim=1)  # (N,)
    
    return wasserstein_dist


def normalize_wasserstein_per_batch(distances: torch.Tensor) -> torch.Tensor:
    """
    Normalize Wasserstein distances to [0, 1] range using min-max scaling per batch.
    
    Args:
        distances: Tensor of shape (N,) or (N, K) - raw Wasserstein distances
        
    Returns:
        Normalized distances in [0, 1] range
    """
    min_dist = distances.min()
    max_dist = distances.max()
    
    # Avoid division by zero
    range_dist = max_dist - min_dist
    if range_dist < 1e-8:
        return torch.zeros_like(distances)
    
    normalized = (distances - min_dist) / range_dist
    return normalized


def compute_feature_discrepancy(
    model,
    clean_images: torch.Tensor,
    aug_images: torch.Tensor,
    device: torch.device,
    epsilon: float = 0.1,
    max_iterations: int = 100
) -> torch.Tensor:
    """
    Compute feature discrepancy score between clean and augmented images using
    Wasserstein distance on model embeddings.
    
    Args:
        model: The model with _encode() method for feature extraction (e.g., ALModel)
        clean_images: Tensor of shape (N, C, H, W) - clean images
        aug_images: Tensor of shape (M, C, H, W) - augmented images
        device: Device to run computations on
        epsilon: Entropic regularization parameter for Sinkhorn
        max_iterations: Maximum Sinkhorn iterations
        
    Returns:
        Tensor of shape (N,) - Wasserstein distance for each clean image to its
        corresponding augmented images
    """
    model.eval()
    with torch.no_grad():
        # Extract features using _encode method (works with ALModel)
        clean_features = model._encode(clean_images.to(device))
        aug_features = model._encode(aug_images.to(device))
    
    # Compute Wasserstein distance
    wasserstein_dist = compute_wasserstein_distance(
        clean_features, aug_features, 
        epsilon=epsilon, 
        max_iterations=max_iterations
    )
    
    return wasserstein_dist



