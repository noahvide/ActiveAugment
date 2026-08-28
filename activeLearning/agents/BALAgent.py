from typing import Dict, List, Optional
import torch
from torch import nn

from activeLearning.agents.BaseAgent import BaseAgent
from models.modelFactory import ALModel

class BALAgent(BaseAgent):
    """
    Implements Balancing Active Learning (BAL).
    - Classic AL: Uses K-Means on the reference pool to find the true Cluster Distance Difference (CDD).
    - Active DA: Uses batch-adapted CDD (distance to own image vs. nearest other image).
    """
    
    requires_reference_features = True

    def get_reference_features(self, model: ALModel, images: torch.Tensor) -> torch.Tensor:
        """Extract features from the labeled pool to serve as K-Means data points."""
        return model._encode(images)
        


    def score_unlabeled(self, 
                        model: ALModel, 
                        unlabeled_batch: torch.Tensor,
                        labeled_features: Optional[torch.Tensor] = None,
                        discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Scores a standard unlabeled batch using the BAL CDD metric.
        """
        unlb_features = model._encode(unlabeled_batch)
        
        if labeled_features is None or labeled_features.size(0) < 2:
            return torch.norm(unlb_features, p=2, dim=1)

        centers = self._get_cluster_centers(labeled_features, num_clusters=10)
        
        dists = torch.cdist(unlb_features, centers, p=2.0)
        
        top2_dists, _ = torch.topk(dists, k=2, dim=1, largest=False)
        d_1st_nearest = top2_dists[:, 0]
        d_2nd_nearest = top2_dists[:, 1]
        
        cdd = torch.abs(d_1st_nearest - d_2nd_nearest)
        
        return -cdd

    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Scores augmentations using batch-adapted CDD.
        """
        if normalized_clean_images is None:
            raise ValueError("BALAgent requires normalized_clean_images to compute the batch-adapted CDD.")
            
        N, K, C, H, W = normalized_candidates.shape
        flat_candidates = normalized_candidates.view(N * K, C, H, W)
        
        clean_features = model._encode(normalized_clean_images)        
        candidate_features = model._encode(flat_candidates).view(N, K, -1)
        
        d_own = torch.norm(candidate_features - clean_features.unsqueeze(1), dim=-1)
        dists = torch.cdist(candidate_features.view(N * K, -1), clean_features, p=2.0).view(N, K, N)
        
        mask = torch.eye(N, device=self.device).view(N, 1, N).expand(N, K, N) > 0.5
        dists.masked_fill_(mask, float('inf'))
        
        d_other_min = dists.min(dim=2).values
        cdd = torch.abs(d_own - d_other_min)
        
        return -cdd
    
    def _get_cluster_centers(self, features: torch.Tensor, num_clusters: int = 10, num_iters: int = 5) -> torch.Tensor:
        N = features.size(0)
        num_clusters = min(num_clusters, N)
        
        indices = torch.randperm(N, device=self.device)[:num_clusters]
        centers = features[indices].clone()
        
        for _ in range(num_iters):
            dists = torch.cdist(features, centers)
            assignments = torch.argmin(dists, dim=1)
            
            new_centers = []
            for i in range(num_clusters):
                cluster_points = features[assignments == i]
                if len(cluster_points) > 0:
                    new_centers.append(cluster_points.mean(dim=0))
                else:
                    new_centers.append(centers[i])
            centers = torch.stack(new_centers)
            
        return centers