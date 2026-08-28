import torch
import torch.nn.functional as F
from torch import nn

from typing import List, Dict, Optional
from activeLearning.agents.BaseAgent import BaseAgent
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from models.modelFactory import ALModel

class TypiClustAugmentationAgent(BaseAgent):
    requires_reference_features = True
    
    def get_reference_features(self, model: ALModel, images: torch.Tensor) -> torch.Tensor: 
        embeddings = model._encode(images)
        return F.normalize(embeddings, p=2.0, dim=1)
    
    def score_unlabeled(self, 
                        model: ALModel, 
                        unlabeled_batch: torch.Tensor, 
                        labeled_features: Optional[torch.Tensor] = None,
                        discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Calculates the inverse typicality of an unlabeled batch against the reference pool.
        """
        unlb_embeddings = self.get_reference_features(model, unlabeled_batch)
        
        ref_embeddings = labeled_features if labeled_features is not None else unlb_embeddings
        
        dist_matrix = torch.cdist(unlb_embeddings, ref_embeddings, p=2.0)
        
        population_size = ref_embeddings.size(0)
        k_nn = max(3, min(int(population_size * 0.10), 20))
        k_nn = min(k_nn, population_size)
        
        if k_nn < 1:
            return torch.zeros(unlb_embeddings.size(0), dtype=torch.float32, device=unlabeled_batch.device)

        knn_distances, _ = torch.topk(dist_matrix, k=k_nn, dim=1, largest=False)
        
        if knn_distances.shape[1] > 1:
            knn_distances = knn_distances[:, 1:]
            
        mean_distance = knn_distances.mean(dim=1)
        typicality = 1.0 / (mean_distance + 1e-5)

        scores = -typicality
        
        return scores


    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Calculates the inverse typicality (scatter) of each candidate pipeline 
        relative to the clean local batch.
        """
        if normalized_clean_images is None:
            raise ValueError("You must provide normalized clean images when using TypiClustAgent in Active DA")
        
        N, K, C, H, W = normalized_candidates.shape
        flat_candidates = normalized_candidates.view(N * K, C, H, W)
        
        clean_features = self.get_reference_features(model, normalized_clean_images)
        
        flat_scores = self.score_unlabeled(
            model=model,
            unlabeled_batch=flat_candidates,
            labeled_features=clean_features,
            discriminator=discriminator
        )
        
        return flat_scores.view(N, K)