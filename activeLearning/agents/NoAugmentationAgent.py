from typing import List, Dict, Optional
import torch
from torch import nn

from activeLearning.agents.BaseAgent import BaseAgent
from configSetup.configModel import ExperimentConfig
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from dataAugementation.augmentations.doNothing import DoNothing
from models.modelFactory import ALModel

class NoAugmentationAgent(BaseAgent):
    """
    Baseline agent for Active DA that strictly applies no augmentations.
    Overrides the augmentation space with DoNothing().
    """
    def __init__(self, 
                 config: ExperimentConfig,
                 agent_seed: int,
                 augmentation_space: Optional[List[BaseAugmentation]] = None,
                 **kwargs):
        super().__init__(config, agent_seed, augmentation_space, **kwargs)
        
        self.k = 1
        self.m = 1
        
        pipeline_length = 1
        if self.config.augmentation_space is not None:
            pipeline_length = self.config.augmentation_space.pipeline_length
             
        self.augmentation_space = [DoNothing() for _ in range(max(1, pipeline_length))]


    def score_unlabeled(self, 
                        model: ALModel, 
                        unlabeled_batch: torch.Tensor,
                        labeled_features: Optional[torch.Tensor] = None,
                        discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Classic AL is intentionally blocked for this agent.
        """
        raise NotImplementedError(
            "NoAugmentationAgent should only be used with Active Augment. "
            "To run a Classic AL baseline without active selection, use the RandomAgent."
        )

    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Returns uniform zero scores since all candidates are mathematically identical 
        """
        N, K = normalized_candidates.shape[:2]
        return torch.zeros((N, K), dtype=torch.float32, device=self.device)