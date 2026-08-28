from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from activeLearning.agents import RandomAgent
from configSetup.configModel import ExperimentConfig
from training.trainers.activeTrainer import ActiveAugmentationTrainer


class MaxUpTrainer(ActiveAugmentationTrainer):
    """
    Implements the MaxUp training strategy.
    For each sample, m augmentations are generated. The loss is computed for all
    m augmentations, and the model's parameters are updated using the gradient
    from only the single, highest-loss ("hardest") augmentation for that sample.
    """

    def __init__(self, 
                 model, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 config: ExperimentConfig, 
                 output_dir: Path, 
                 run: int, 
                 aug_service, 
                 budget: int):
        # MaxUp is not an "active" selection strategy. It generates m random augmentations.
        agent = RandomAgent(config=config, agent_seed=config.dataset.seed + run)
        super().__init__(model, train_loader, val_loader, aug_service, agent, config, output_dir, run, budget)
        self.m = self.config.augmentation_strategy.m_select

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.training.lr
        )
    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        aug_stats = {}

        for raw_images, labels in self.train_loader:
            self.optimizer.zero_grad()

            raw_images, labels = raw_images.to(self.device), labels.to(self.device)
            normalized_clean_images = self.norm_transform(raw_images)

            # Generate/select m augmentations per image
            aug_images, aug_metadata, aug_labels = self._get_augmentations(raw_images, labels, normalized_clean_images)

            N = normalized_clean_images.size(0)

            # Forward pass for all augmentations at once
            # aug_images is already (N*m, C, H, W)
            outputs = self.model(aug_images)

            # Calculate loss for each augmentation individually, without reduction
            # The criterion needs to be re-instantiated to change reduction
            individual_losses = torch.nn.CrossEntropyLoss(reduction='none')(outputs, aug_labels)

            # Reshape losses to (N, m) to find the max for each original sample
            losses_reshaped = individual_losses.view(N, self.m)

            # Find the maximum loss and its index for each sample
            max_losses, max_indices = torch.max(losses_reshaped, dim=1)

            # The final loss for the backward pass is the sum of the maximum losses
            # This ensures the gradient is computed only from the "hardest" augmentations
            final_loss = max_losses.sum()

            final_loss.backward()
            self.optimizer.step()

            # For logging, we use the sum of max losses
            total_loss += final_loss.item()
            self.global_step += 1
            self._update_aug_stats(aug_stats, aug_metadata)

        avg_train_loss = total_loss / len(self.train_loader.dataset) # type: ignore
        return avg_train_loss, aug_stats
