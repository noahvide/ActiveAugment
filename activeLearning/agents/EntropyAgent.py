from typing import List, Dict, Optional

import torch
from torch import nn

from activeLearning.agents.BaseAgent import BaseAgent
from models.modelFactory import ALModel


class EntropyAgent(BaseAgent):

    def score_unlabeled(self, 
                        model: ALModel, 
                        unlabeled_batch: torch.Tensor, 
                        labeled_features: Optional[torch.Tensor] = None,
                        discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        logits = model(unlabeled_batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
        eps = 1e-7
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
        return entropy

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