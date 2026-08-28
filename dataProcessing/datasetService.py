import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Optional
import numpy as np

from models.modelFactory import ALModel
from activeLearning.agents.BaseAgent import BaseAgent
from dataProcessing.datasets.cachedDataset import CachedDataset

class DatasetService:
    def __init__(self, full_dataset, device: str):
        """
        Manages the state of labeled vs. unlabeled data across all framework modes.
        """
        self.full_dataset = full_dataset
        self.device = torch.device(device)
        
        self.labeled_mask = torch.zeros(len(full_dataset), dtype=torch.bool)
        
    @property
    def current_labeled_size(self) -> int:
        return int(self.labeled_mask.sum().item())
        
    @property
    def current_unlabeled_size(self) -> int:
        return int((~self.labeled_mask).sum().item())

    def initialize_from_indices(self, indices: List[int]):
        """
        Forces the labeled pool to exactly match a pre-computed list of indices.
        """
        self.labeled_mask.zero_()
        self.labeled_mask[indices] = True
        
    def acquire_random_stratified(self, num_to_acquire: int, seed: int):
        rng = np.random.default_rng(seed)
        unlabeled_indices = (~self.labeled_mask).nonzero(as_tuple=True)[0].cpu().numpy()
        
        if num_to_acquire > len(unlabeled_indices):
            raise ValueError(f"Cannot acquire {num_to_acquire} samples. Only {len(unlabeled_indices)} available.")
            
        all_targets = np.array(self.full_dataset.targets)

        unlabeled_targets = all_targets[unlabeled_indices]
        classes, class_counts = np.unique(unlabeled_targets, return_counts=True)
        
        class_proportions = class_counts / len(unlabeled_indices)
        samples_per_class = np.floor(class_proportions * num_to_acquire).astype(int)
        
        remainder = num_to_acquire - samples_per_class.sum()
        if remainder > 0:
            for _ in range(remainder):
                available_capacity = class_counts - samples_per_class
                valid_class_indices = np.where(available_capacity > 0)[0]
                
                if len(valid_class_indices) == 0:
                    break
                    
                chosen_idx = rng.choice(valid_class_indices)
                samples_per_class[chosen_idx] += 1

        selected_indices = []
        for cls, num_samples in zip(classes, samples_per_class):
            if num_samples == 0:
                continue
                
            cls_mask = (unlabeled_targets == cls)
            cls_indices = unlabeled_indices[cls_mask]
            
            selected_cls_indices = rng.choice(cls_indices, size=num_samples, replace=False)
            selected_indices.extend(selected_cls_indices)
            
        self.labeled_mask[selected_indices] = True

    def acquire_labels(self, model: ALModel, agent: BaseAgent, num_to_acquire: int):
        """
        Scores the unlabeled pool using the Agent and moves the top candidates to the labeled pool.
        """
        if num_to_acquire <= 0:
            return
            
        model.eval()
        
        unlabeled_indices = (~self.labeled_mask).nonzero(as_tuple=True)[0]
        labeled_indices = self.labeled_mask.nonzero(as_tuple=True)[0]
        
        if len(unlabeled_indices) < num_to_acquire:
            raise ValueError(f"Agent requested {num_to_acquire} images, but only {len(unlabeled_indices)} remain.")
        
        unlabeled_subset = Subset(self.full_dataset, unlabeled_indices.tolist())
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=128, shuffle=False, num_workers=0)
        
        labeled_features = None
        if agent.requires_reference_features and len(labeled_indices) > 0: 
            labeled_subset = Subset(self.full_dataset, labeled_indices.tolist())
            labeled_loader = DataLoader(labeled_subset, batch_size=128, shuffle=False, num_workers=0)
            
            features_list = []
            with torch.no_grad():
                for imgs, _ in labeled_loader:
                    feats = agent.get_reference_features(model, imgs.to(self.device)) 
                    features_list.append(feats)
            labeled_features = torch.cat(features_list)

        all_scores = []
        with torch.no_grad():
            for imgs, _ in unlabeled_loader:
                imgs = imgs.to(self.device)
                
                scores = agent.score_unlabeled(model, imgs, labeled_features=labeled_features)
                all_scores.append(scores.cpu())
                
        all_scores = torch.cat(all_scores)
        
        _, top_k_local_indices = torch.topk(all_scores, num_to_acquire, largest=True)
        
        selected_global_indices = unlabeled_indices[top_k_local_indices]
        self.labeled_mask[selected_global_indices] = True
        

    def get_train_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Returns a ready-to-train DataLoader containing only the labeled pool."""
        labeled_indices = self.labeled_mask.nonzero(as_tuple=True)[0].tolist()
                
        subset = Subset(self.full_dataset, labeled_indices)
        
        cached_subset = CachedDataset(subset)
        
        return DataLoader(
            cached_subset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    
    def get_unlabeled_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> Optional[DataLoader]:
        """
        Returns a DataLoader containing only the unlabeled pool. 
        """
        unlabeled_indices = (~self.labeled_mask).nonzero(as_tuple=True)[0].tolist()
        
        if len(unlabeled_indices) == 0:
            return None
                
        subset = Subset(self.full_dataset, unlabeled_indices)
        
        cached_subset = CachedDataset(subset)
        
        return DataLoader(
            cached_subset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            drop_last=True
        )