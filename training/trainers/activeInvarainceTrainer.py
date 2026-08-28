from email.headerregistry import DateHeader
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

from activeLearning.agents import BaseAgent, NoAugmentationAgent
from configSetup.configModel import ExperimentConfig
from configSetup.configEnums import AugmentationStrategyType
from dataAugementation.augmentationService import AugmentationService
from models.modelFactory import ALModel
from training.trainers.baseTrainer import BaseTrainer

class ActiveInvarianceTrainer(BaseTrainer):
    def __init__(self, 
                 model: ALModel, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 config: ExperimentConfig, 
                 output_dir: Path, 
                 run: int, 
                 aug_service: AugmentationService,
                 agent: Optional[BaseAgent] = None,
                 budget: Optional[int] = None):
        super().__init__(model, train_loader, val_loader, config, output_dir, run, budget)
        
        self.agent = agent
        self.aug_service = aug_service
        
        self.lambda_con = getattr(config.training, 'lambda_con', 0.1) 
        self.temperature = getattr(config.training, 'temperature', 0.07)        
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.training.lr
        )
        
        self.global_step = 0

    def _compute_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Implements L_con: Supervised Contrastive Loss.
        features: (2B, d) - combined clean and augmented features.
        labels: (2B) - combined labels.
        """
        device = features.device
        batch_size = labels.shape[0] # 2B
        
        features = F.normalize(features, dim=1)
        
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        return -mean_log_prob_pos.mean()

    def _train_epoch(self) -> tuple[float, dict]:
        self.model.train()
        total_loss = 0.0
        aug_stats = {}
        self.aug_service.set_normalizer(self.norm_transform)

        for raw_images, labels in self.train_loader:
            raw_images, labels = raw_images.to(self.device), labels.to(self.device)            
            normalized_clean_images = self.norm_transform(raw_images)
            
            if self.config.augmentation_strategy.type != AugmentationStrategyType.NONE:
                selected_aug, selected_metadata, aug_labels = self._get_augmentations(raw_images, labels, normalized_clean_images)
                final_images, final_labels = self._integrate_augmentations(normalized_clean_images, selected_aug, labels, aug_labels)
            else:
                final_images, final_labels = normalized_clean_images, labels
                selected_metadata = []
            
            features = self.model._encode(final_images)
            logits = self.model._classify(features)
            
            l_sup = self.criterion(logits, final_labels)

            l_con = self._compute_contrastive_loss(features, final_labels)

            loss = l_sup + (self.lambda_con * l_con)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1
            self._update_aug_stats(aug_stats, selected_metadata)

        return total_loss / len(self.train_loader), aug_stats
