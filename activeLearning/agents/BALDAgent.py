import torch
import torch.nn.functional as F
from torch import nn

from typing import List, Dict, Optional
from activeLearning.agents.BaseAgent import BaseAgent
from configSetup.configModel import ExperimentConfig
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from models.modelFactory import ALModel


class BALDAgent(BaseAgent):
    """
    Calculates the BALD (Mutual Information) score for each candidate pipeline 
    using Monte Carlo Dropout.
    """
    def __init__(self, 
                 config: ExperimentConfig,
                 agent_seed: int,
                 augmentation_space: List[BaseAugmentation],
                 **kwargs):
        super().__init__(config, agent_seed, augmentation_space, **kwargs)
        self.dropout_trials = 10

    def score_unlabeled(self, 
                        model: ALModel, 
                        unlabeled_batch: torch.Tensor,
                        labeled_features: Optional[torch.Tensor] = None,
                        discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Scores a standard unlabeled batch using MC Dropout to find Mutual Information.
        """
        
        model.train()
        has_dropout = False
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()
            if isinstance(m, torch.nn.Dropout):
                has_dropout = True
                
        if not has_dropout:
            model.eval()
            raise ValueError("BALDAgent requires the model to have dropout layers for Monte Carlo Dropout.")
        
        preds_collection = []
        
        with torch.no_grad():
            for _ in range(self.dropout_trials):
                logits = model(unlabeled_batch, stochastic=True)
                probs = F.softmax(logits, dim=1)
                preds_collection.append(probs)
                
        preds = torch.stack(preds_collection, dim=1)
        
        mean_preds = preds.mean(dim=1) 
        entropy_of_mean = -torch.sum(mean_preds * torch.log(mean_preds + 1e-9), dim=1)
        
        entropy_per_trial = -torch.sum(preds * torch.log(preds + 1e-9), dim=2)
        mean_of_entropies = entropy_per_trial.mean(dim=1)
        
        bald_scores = entropy_of_mean - mean_of_entropies
        
        model.eval()
        
        return bald_scores


    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:

        N, K, C, H, W = normalized_candidates.shape
        flat_candidates = normalized_candidates.view(N * K, C, H, W)
        
        flat_scores = self.score_unlabeled(model, flat_candidates, discriminator=discriminator)
        
        return flat_scores.view(N, K)