from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configSetup.configModel import ExperimentConfig
from models.modelFactory import ALModel
from training.trainers.baseTrainer import BaseTrainer


class StandardTrainer(BaseTrainer):
    def __init__(self, 
                 model: ALModel, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 config: ExperimentConfig, 
                 output_dir: Path, 
                 run: int, 
                 budget=None):
        super().__init__(model, train_loader, val_loader, config, output_dir, run, budget)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.training.lr
        )

    def _train_epoch(self) -> tuple[float, dict]:
        self.model.train()
        total_loss = 0
        
        for raw_images, labels in self.train_loader:
            raw_images, labels = raw_images.to(self.device), labels.to(self.device)
            normalized_images = self.norm_transform(raw_images)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(normalized_images), labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader), {}
    
