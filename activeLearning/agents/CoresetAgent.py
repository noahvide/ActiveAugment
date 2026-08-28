from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from activeLearning.agents.BaseAgent import BaseAgent
from models.modelFactory import ALModel


class CoresetAgent(BaseAgent):

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
        Scores a standard unlabeled batch based on Coreset distance.
        """
        unlb_embeddings = self.get_reference_features(model, unlabeled_batch)

        if labeled_features is not None:
            dist_matrix = torch.cdist(unlb_embeddings, labeled_features, p=2.0)
            scores, _ = torch.min(dist_matrix, dim=1)
        else:
            scores = torch.norm(unlb_embeddings, p=2, dim=1)
            
        return scores
    
    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Calculates the Core-set distance for each candidate pipeline relative 
        to the unaugmented local batch.
        """
        if normalized_clean_images is None:
            raise ValueError("You must provide normalized clean images when using CoresetAgent in Active DA")
            
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
