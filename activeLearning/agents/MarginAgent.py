import torch
import torch.nn.functional as F
from torch import nn

from typing import List, Dict, Optional
from activeLearning.agents.BaseAgent import BaseAgent
from models.modelFactory import ALModel

class MarginAgent(BaseAgent):

    def score_unlabeled(self, 
                    model: ALModel, 
                    unlabeled_batch: torch.Tensor, 
                    labeled_features: Optional[torch.Tensor] = None,
                    discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        logits = model(unlabeled_batch)
        probs = F.softmax(logits, dim=1)
        
        top_2_probs, _ = probs.topk(2, dim=1)
        
        margin = top_2_probs[:, 0] - top_2_probs[:, 1]
        
        scores = 1.0 - margin
        return scores

    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:

        N, K, C, H, W = normalized_candidates.shape        
        flat_candidates = normalized_candidates.view(N * K, C, H, W)
        
        flat_scores = self.score_unlabeled(model, flat_candidates, discriminator=discriminator)
        
        scores = flat_scores.view(N, K)
        
        return scores