from typing import Dict, List, Optional
import torch
from torch import nn

from activeLearning.agents.BaseAgent import BaseAgent
from configSetup.configModel import ExperimentConfig
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from models.modelFactory import ALModel

class RandomAgent(BaseAgent):
    """
    Selects random candidates/images by assigning uniformly random scores.
    """
    def __init__(self, 
                 config: ExperimentConfig, 
                 agent_seed: int,
                 augmentation_space: Optional[List[BaseAugmentation]] = None, 
                 **kwargs):
        super().__init__(config, agent_seed, augmentation_space, **kwargs)
        
        self.penalty_lambda = 0.0 


    def score_unlabeled(self, 
                        model: ALModel, 
                        unlabeled_batch: torch.Tensor,
                        labeled_features: Optional[torch.Tensor] = None,
                        discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """Assigns a random float [0.0, 1.0) to each unlabeled image."""
        B = unlabeled_batch.size(0)
        return torch.rand((B,), generator=self._rng, device='cpu').to(self.device)

    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """Assigns a random float [0.0, 1.0) to each augmented candidate."""
        N, K = normalized_candidates.shape[:2]
        return torch.rand((N, K), generator=self._rng, device='cpu').to(self.device)