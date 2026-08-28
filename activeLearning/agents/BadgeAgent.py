import torch
import torch.nn.functional as F
from torch import nn

from typing import List, Dict, Optional
from activeLearning.agents.BaseAgent import BaseAgent
from models.modelFactory import ALModel

class BadgeAgent(BaseAgent):
    """
    Calculates the BADGE score by measuring the distance of each candidate's 
    gradient embedding to the gradient embeddings of the clean batch.
    """
    
    requires_reference_features = True

    def get_reference_features(self, model: ALModel, images: torch.Tensor) -> torch.Tensor:
        """
        Helper method to compute the BADGE gradient embeddings:
        g = (p - y_hat) x features
        """
        features = model._encode(images)
        logits = model._classify(features)
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        y_hat = F.one_hot(preds, num_classes=logits.size(-1)).float()
        
        scale_factor = probs - y_hat
        
        grad_embeddings = torch.einsum('bd,bc->bdc', features, scale_factor)
        
        return grad_embeddings.reshape(images.size(0), -1)
    
    def score_unlabeled(self, 
                        model: ALModel, 
                        unlabeled_batch: torch.Tensor, 
                        labeled_features: Optional[torch.Tensor] = None,
                        discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Scores a standard unlabeled batch. 
        Requires `labeled_features` to be the gradient embeddings of the labeled set.
        """
        unlb_grad_embeddings = self.get_reference_features(model, unlabeled_batch)

        if labeled_features is not None:
            dist_matrix = torch.cdist(unlb_grad_embeddings, labeled_features, p=2.0)
            scores, _ = torch.min(dist_matrix, dim=1)
        else:
            scores = torch.norm(unlb_grad_embeddings, p=2, dim=1)
            
        return scores
        

    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        
        if normalized_clean_images is None:
            raise ValueError("You must provide normalized clean images when using BadgeAgent")
        N, K, C, H, W = normalized_candidates.shape
        
        flat_candidates = normalized_candidates.view(N * K, C, H, W)
        
        clean_grad_embeddings = self.get_reference_features(model, normalized_clean_images)
        
        min_dists = self.score_unlabeled(
            model=model, 
            unlabeled_batch=flat_candidates, 
            labeled_features=clean_grad_embeddings,
            discriminator=discriminator
        )
        
        scores = min_dists.view(N, K)
        
        return scores