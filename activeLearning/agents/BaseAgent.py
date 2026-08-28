from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F

from configSetup.configModel import ExperimentConfig
from configSetup.configEnums import AugmentationStrategyType
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from models.modelFactory import ALModel

class BaseAgent(ABC):
    def __init__(self, 
                 config: ExperimentConfig,
                 agent_seed: int,
                 augmentation_space: Optional[List[BaseAugmentation]] = None,
                 **kwargs):

        self.agent_seed: int = agent_seed
        self._rng = torch.Generator(device='cpu')
        self._rng.manual_seed(agent_seed)
        
        self.config: ExperimentConfig = config
        self.device = self.config.training.device
        self.augmentation_space = augmentation_space
        self.requires_reference_features = False
        self.is_active_augment = (self.config.augmentation_strategy.type == AugmentationStrategyType.ACTIVE)
        
        if self.is_active_augment:
            aug_strat = self.config.augmentation_strategy
            augs_space = self.config.augmentation_space
            
            k_conf = aug_strat.k_candidates
            self.k = len(augmentation_space) if augmentation_space and not k_conf else k_conf
            
            self.m = aug_strat.m_select
            
            self.fixed_pipeline_length = augs_space.fixed_pipeline_length if augs_space else True
            
            # Feature discrepancy scoring
            if aug_strat.discrepancy is not None:
                self.enable_feature_discrepancy = True
                self.normalize_discrepancy = aug_strat.discrepancy.normalize
            else:
                self.enable_feature_discrepancy = False
        else:
            self.k, self.m = 0, 0
            self.fixed_pipeline_length = True
            self.enable_feature_discrepancy = False

        self.aug_scores = []
        self.history = []
        self.step_counter = 0
        self._model_primed = False

    def predict(self, 
                state_model: ALModel, 
                normalized_candidates: torch.Tensor, 
                metadata: List[List[Dict]],
                normalized_clean_images: Optional[torch.Tensor] = None,
                discriminator: Optional[nn.Module] = None,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Scores the (N, K, C, H, W) candidates and selects the top M per sample.
        """
        state_model.eval()
        N, K, C, H, W = normalized_candidates.shape
        
        with torch.no_grad():
            if not self._model_primed and normalized_clean_images is not None:
                _ = state_model(normalized_clean_images[0:1].to(self.device))
                self._model_primed = True
                
            scores = self._score_batch(state_model, normalized_candidates, metadata, normalized_clean_images, discriminator)
            # Compute feature discrepancy and multiply with scores
            if self.enable_feature_discrepancy and normalized_clean_images is not None:
                d_ik = self._compute_feature_discrepancy(
                    state_model, normalized_candidates, normalized_clean_images, labels
                )
                
                scores = scores * d_ik
            
            _, top_indices = torch.topk(scores, self.m, dim=1)
        selected_images = []
        selected_metadata = []
        
        for i in range(N):
            for j in range(self.m):
                chosen_idx = int(top_indices[i, j].item())
                selected_images.append(normalized_candidates[i, chosen_idx])
                
                meta = metadata[i][chosen_idx].copy()
                meta["reward"] = scores[i, chosen_idx].item()
                selected_metadata.append(meta)
                
        self._log_step(scores, selected_metadata)
        state_model.train()

        return torch.stack(selected_images), selected_metadata

    def _log_step(self, scores: Optional[torch.Tensor], selected_metadata: List[Dict]):
        """Records agent decisions for post-run analysis."""
        if scores is not None:
            mean_batch_score = scores.mean().item()
            mean_std_dev = (scores.std(dim=1, unbiased=False) + 1e-8).mean().item() 
            
            top_scores = scores.max(dim=1).values
            mean_scores = scores.mean(dim=1)
            mean_margin = (top_scores - mean_scores).mean().item()
            
            max_score = top_scores.mean().item()
            min_score = scores.min(dim=1).values.mean().item()

            self.aug_scores.append({
                "step": self.step_counter,
                "mean_batch_score": mean_batch_score,
                "mean_std_dev": mean_std_dev,
                "mean_margin": mean_margin,
                "max_score": max_score,
                "min_score": min_score
            })
        
        selected_augs = []
        selected_intensities = []
        
        for meta_wrapper in selected_metadata:
            pipeline = meta_wrapper.get("pipeline", [])
            
            pipeline_augs = [step.get("augmentation", "Unknown") for step in pipeline]
            pipeline_intensities = [step.get("intensity", "N/A") for step in pipeline]
            
            selected_augs.append(pipeline_augs)
            selected_intensities.append(pipeline_intensities)
            
        self.history.append({
            "step": self.step_counter,
            "selected_augmentations": selected_augs,
            "selected_intensities": selected_intensities
        })
        self.step_counter += 1

    @abstractmethod
    def score_unlabeled(self, 
                        model: ALModel, 
                        unlabeled_batch: torch.Tensor,
                        labeled_features: Optional[torch.Tensor] = None,
                        discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Evaluates a standard batch of unlabeled images.
        
        Args:
            model: The current model state.
            unlabeled_batch: Tensor of shape (B, C, H, W).
            labeled_features: Optional tensor of existing dataset features.
            discriminator: Optional discriminator module for adversarial scoring.
        Returns:
            torch.Tensor: A 1D tensor of float scores of shape (B,).
        """
        pass
    
    @abstractmethod
    def _score_batch(self, 
                     model: ALModel, 
                     normalized_candidates: torch.Tensor, 
                     metadata: List[List[Dict]],
                     normalized_clean_images: Optional[torch.Tensor] = None,
                     discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Evaluates a batch of augmented candidates.
        
        Args:
            model: The current model state.
            normalized_candidates: Tensor of shape (N, K, C, H, W)
            metadata: List of length N containing lists of length K with augmentation details.
            normalized_clean_images: Optional tensor of shape (N, C, H, W)
            discriminator: Optional discriminator module for adversarial scoring.

        Returns:
            torch.Tensor: A tensor of float scores of shape (N, K).
        """
        raise NotImplementedError("Each agent must implement its own batch-aware scoring logic!")

    def _compute_feature_discrepancy(
        self,
        state_model: ALModel,
        normalized_candidates: torch.Tensor,
        normalized_clean_images: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.enable_feature_discrepancy or normalized_clean_images is None:
            raise ValueError("Feature discrepancy is disabled or clean images are missing.")
        
        N, K, C, H, W = normalized_candidates.shape
        
        with torch.no_grad():
            clean_features = state_model._encode(normalized_clean_images.to(self.device))
            flat_cands = normalized_candidates.view(-1, C, H, W).to(self.device)
            cand_features = state_model._encode(flat_cands).view(N, K, -1)

        clean_ext = clean_features.unsqueeze(1).expand(-1, K, -1)
        cos_sim = F.cosine_similarity(clean_ext, cand_features, dim=2)
        d_ik = (1.0 - cos_sim) / 2.0

        if self.normalize_discrepancy:
            d_bar_k = d_ik.mean(dim=0, keepdim=True)
            discrepancy = d_ik / (d_bar_k + 1e-8)
        else:
            discrepancy = d_ik

        return discrepancy
        
    def get_reference_features(self, model: ALModel, images: torch.Tensor) -> torch.Tensor: # type: ignore
        """
        Optional method to compute reference features from the labeled set (used by Coreset and BADGE).
        """
        pass
    