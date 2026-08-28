from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader

from activeLearning.agents import BaseAgent, NoAugmentationAgent
from configSetup.configModel import ExperimentConfig
from configSetup.configEnums import AugmentationStrategyType
from dataAugementation.augmentationService import AugmentationService
from models.modelFactory import ALModel
from training.trainers.baseTrainer import BaseTrainer


class ActiveAugmentationTrainer(BaseTrainer):
    def __init__(self, 
                 model: ALModel, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 aug_service: AugmentationService, 
                 agent: Optional[BaseAgent], 
                 config: ExperimentConfig, 
                 output_dir: Path, 
                 run: int, 
                 budget: int):
        if config.augmentation_strategy.type not in [AugmentationStrategyType.ACTIVE, AugmentationStrategyType.STATIC]:
            raise ValueError("ActiveAugmentationTrainer requires 'active' or 'static' block in config.")
        if not config.augmentation_space:
            raise ValueError("ActiveAugmentationTrainer requires an 'augmentation_space' block in config.")        
        super().__init__(model, train_loader, val_loader, config, output_dir, run, budget)
        
        self.aug_service = aug_service
        self.agent = agent
        self.norm_transform = self.train_loader.dataset.norm_transform #type: ignore
        self.aug_service.set_normalizer(self.norm_transform)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.training.lr
        )
        

    def _train_epoch(self) -> Tuple[float, dict]:
        self.model.train()
        total_loss = 0
        aug_stats = {}
        
        for raw_images, labels in self.train_loader:
            raw_images, labels = raw_images.to(self.device), labels.to(self.device)
            normalized_clean_images = self.norm_transform(raw_images)
            if self.config.augmentation_strategy.type != AugmentationStrategyType.NONE:
                selected_aug, selected_metadata, aug_labels = self._get_augmentations(raw_images, labels, normalized_clean_images)
                final_images, final_labels = self._integrate_augmentations(normalized_clean_images, selected_aug, labels, aug_labels)
            else:
                final_images, final_labels = normalized_clean_images, labels
                selected_metadata = []
                
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(final_images), final_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            self.global_step += 1
            self._update_aug_stats(aug_stats, selected_metadata)
            
        avg_train_loss = total_loss / len(self.train_loader)
        return avg_train_loss, aug_stats