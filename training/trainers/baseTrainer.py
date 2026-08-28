from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from activeLearning.agents import BaseAgent
from configSetup.configModel import ExperimentConfig
from configSetup.configEnums import AugmentationStrategyType, CandidateGenerationMode
from configSetup.enums import MetricName
from dataAugementation.augmentationFactory import augmentation_factory
from dataAugementation.augmentationService import AugmentationService
from dataProcessing.datasets.test_dataset import TestDataset
from eval_utils import get_test_scores
from models.modelFactory import ALModel
from training.metricFactory import MetricFactory


class BaseTrainer(ABC):
    def __init__(self, 
                 model: ALModel, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 config: ExperimentConfig, 
                 output_dir: Path, 
                 run: int, 
                 budget=None):
        self.device = torch.device(config.training.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.run = run
        self.budget = budget
        self.current_epoch = 0
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.norm_transform = self.train_loader.dataset.norm_transform # type: ignore
        
        self.history = []
        
        self.num_classes = len(train_loader.dataset.classes) # type: ignore
        
        metric_names = config.training.metrics
        self.val_metrics = MetricFactory.get_metrics(metric_names, self.num_classes, self.device)

        self.monitor_metric = config.training.monitor
        if self.monitor_metric not in metric_names and self.monitor_metric != "val_loss":
            raise ValueError(f"Monitoring '{self.monitor_metric}' but it is not in metrics list.")
            
        self.best_metric_val = -float('inf') if self.monitor_metric != "val_loss" else float('inf')
        self.best_monitor_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_loss_epoch = 0
        self.best_train_loss = float('inf')
        self.best_train_loss_epoch = 0


        self.use_annealing = config.augmentation_strategy.use_annealing
        self.total_steps = config.training.active_epochs * len(train_loader)
        self.global_step = 0

    @abstractmethod
    def _train_epoch(self) -> tuple[float, dict]:
        """Must be implemented by subclasses."""
        pass
        
    def _get_augmentations(self, raw_images: torch.Tensor, labels: torch.Tensor, normalized_clean_images: torch.Tensor, discriminator: Optional[torch.nn.Module] = None):
        """Generates/Selects augmentations based on the configured strategy (Active vs Static)."""
        if not self.config.augmentation_space:
            raise ValueError("Augmentation space not found.")
        
        aug_service: Optional[AugmentationService] = getattr(self, "aug_service", None)
        if not aug_service:
            raise ValueError("Augmentation service not found.")
        
        aug_pipeline = augmentation_factory(self.config.augmentation_space.augmentations, self.config.augmentation_space.fixed_pipeline_length, self.config.dataset.name)
        agent : Optional[BaseAgent] = getattr(self, "agent", None)
        
        if self.config.augmentation_strategy.type == AugmentationStrategyType.STATIC:                
            aug_images, batch_metadata = aug_service.apply_pipeline(raw_images, aug_pipeline)
            flat_metadata = [item[0] for item in batch_metadata] if batch_metadata else []
            return aug_images.to(self.device), flat_metadata, labels
        elif agent is None:
            raise ValueError("Agent not found.")
            
        pi_t = self._get_annealed_probs() if self.use_annealing else None
        
        
        if self.config.augmentation_strategy.candidate_mode == CandidateGenerationMode.INTENSITY_SWEEP:
            candidates, batch_metadata = aug_service.generate_intensity_sweeps(raw_images=raw_images, rng=agent._rng, pi_t=pi_t)
        else:
            candidates, batch_metadata = aug_service.generate_random_pipelines(raw_images=raw_images, k=self.config.augmentation_strategy.k_candidates, pipeline_length=self.config.augmentation_space.pipeline_length, rng=agent._rng, fixed_pipeline_length=self.config.augmentation_space.fixed_pipeline_length)
            
        self.model.eval()
        selected_aug_images, selected_metadata = agent.predict(
            state_model=self.model, normalized_candidates=candidates.to(self.device), 
            metadata=batch_metadata, normalized_clean_images=normalized_clean_images, discriminator=discriminator, labels=labels
        )
        self.model.train()
                        
        m = selected_aug_images.size(0) // normalized_clean_images.size(0)
        aug_labels = labels.repeat_interleave(m) if m > 1 else labels
        return selected_aug_images, selected_metadata, aug_labels

    def _integrate_augmentations(self, clean_images: torch.Tensor, aug_images: torch.Tensor, clean_labels: torch.Tensor, aug_labels: torch.Tensor):
        """Merges original images with augmentations using 'concat' or 'fractional' mode."""
        mode = self.config.augmentation_strategy.integration_mode
        if mode == "concat":
            return torch.cat([clean_images, aug_images], dim=0), torch.cat([clean_labels, aug_labels], dim=0)
        elif mode == "fractional":
            B = clean_images.size(0)
            num_replace = int(B * self.config.augmentation_strategy.integration_fraction)
            final_images, final_labels = clean_images.clone(), clean_labels.clone()
            if num_replace > 0:
                indices = torch.randperm(B, device=self.device)[:num_replace]
                final_images[indices], final_labels[indices] = aug_images[indices], aug_labels[indices]
            return final_images, final_labels
        else:
            raise NotImplementedError(f"Integration mode '{mode}' not implemented.")


    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0

        for raw_images, labels in self.val_loader:
            raw_images, labels = raw_images.to(self.device), labels.to(self.device)
            normalized_images = self.norm_transform(raw_images)
            outputs = self.model(normalized_images)
            
            total_loss += self.criterion(outputs, labels).item()
            
            self.val_metrics.update(outputs, labels)
        
        computed_metrics = self.val_metrics.compute()
        
        results = {k if isinstance(k, str) else k.value: v.item() for k, v in computed_metrics.items()}        
        val_loss_key = MetricName.VAL_LOSS.value
        results[val_loss_key] = total_loss / len(self.val_loader)
                
        return results

    @torch.no_grad()
    def evaluate(self):
        test_ds = TestDataset(root_dir=self.config.dataset.name, 
                              weights_status=self.config.model.weights_status)
        
        checkpoint_pth = self.output_dir / f"best_{self.monitor_metric.value}_model.pth"
        test_scores_monitor = get_test_scores(model=self.model, 
                                               checkpoint_path=checkpoint_pth, 
                                               test_ds=test_ds, 
                                               device=self.device)
        
        with open(Path(self.output_dir) / f"test_scores_best_{self.monitor_metric.value}_model.json", 'w') as f:
            json.dump(test_scores_monitor, f, indent=4)

    def fit(self):
        epochs = self.config.training.active_epochs
        
        
        final_val_score = 0.0
        
        epoch_pbar = tqdm(range(epochs), desc=f"Budget {self.budget} Training")
        
        for _ in epoch_pbar:

            epoch = self.current_epoch
            epoch_pbar.set_description(f"Epoch {epoch}/{epochs} (budget: {self.budget})")

            train_loss, aug_stats = self._train_epoch()
                
            val_results = self._validate()
            
            epoch_log = {
                "epoch": epoch, 
                "train_loss": train_loss, 
                **val_results,
            }
            self.history.append(epoch_log)
            pd.DataFrame(self.history).to_csv(self.output_dir / "metrics.csv", index=False)   
            
            aug_log_entry = {
                "epoch": epoch,
                "budget": self.budget,
                "run": self.run,
                "stats": aug_stats
            }  
            with open(self.output_dir / "augmentation_stats.jsonl", "a") as f:
                f.write(json.dumps(aug_log_entry) + "\n")
                
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.best_train_loss_epoch = epoch
            
            current_val = val_results.get(self.monitor_metric)
            if current_val is None:
                raise RuntimeError(f"Could not determine the result for the monitored metric during training. Monitored metric: {self.monitor_metric}. Got {current_val}")
            
            final_val_score = current_val
            
            is_best_monitor = False
            if self.monitor_metric == MetricName.VAL_LOSS:
                if current_val < self.best_metric_val:
                    self.best_metric_val = current_val
                    is_best_monitor = True
            else:
                if current_val > self.best_metric_val:
                    self.best_metric_val = current_val
                    is_best_monitor = True

            val_loss = val_results[MetricName.VAL_LOSS]
            is_best_val_loss = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best_val_loss = True
                self.best_val_loss_epoch = epoch


            if is_best_monitor:
                self.best_monitor_epoch = epoch
                self.checkpoint_path = self.output_dir / f"best_{self.monitor_metric.value}_model.pth"
                torch.save(self.model.state_dict(), self.checkpoint_path)

            if is_best_val_loss or is_best_monitor:
                best_metrics = {
                    MetricName.VAL_LOSS.value: {
                        "epoch": self.best_val_loss_epoch,
                        "value": self.best_val_loss
                    },
                    self.monitor_metric.value: {
                        "epoch": self.best_monitor_epoch,
                        "value": self.best_metric_val
                    }
                }
                
                with open(self.output_dir / f"best_metrics.json", "w") as f:
                    json.dump(best_metrics, f, indent=4)

            
            best_val_epoch = self.best_val_loss_epoch
            best_train_epoch = self.best_train_loss_epoch
            best_f1_epoch = self.best_monitor_epoch
            
            epoch_pbar.set_postfix({
                'Train L': f"{train_loss:.4f} (Best: {self.best_train_loss:.4f} @ Ep{best_train_epoch})",
                'Val L': f"{val_loss:.4f} (Best: {self.best_val_loss:.4f} @ Ep{best_val_epoch})",
                f"{self.monitor_metric.value}": f"{current_val:.4f} (Best: {self.best_metric_val:.4f} @ Ep{best_f1_epoch})"
            })
            
            self.current_epoch += 1
            
        epoch_pbar.close()
        best_val_epoch = self.best_val_loss_epoch
        best_train_epoch = self.best_train_loss_epoch
        best_f1_epoch = self.best_monitor_epoch
        print(f"Best Train Loss: {self.best_train_loss:.4f} @ Ep{best_train_epoch}")
        print(f"Best Val Loss: {self.best_val_loss:.4f} @ Ep{best_val_epoch}")
        print(f"Best {self.monitor_metric.value}: {self.best_metric_val:.4f} @ Ep{best_f1_epoch}")
        return final_val_score

    def _update_aug_stats(self, stats: dict, metadata: list):
        """Helper to track which augmentations are being selected."""
        for meta in metadata:
            pipeline = meta.get("pipeline", [])
            for step in pipeline:
                name = step.get("augmentation", "Unknown")
                if name in ["N/A", "DoNothing"]: continue
                entry = stats.setdefault(name, {"applied_count": 0})
                entry["applied_count"] += 1
    
    def _get_annealed_probs(self) -> Tuple[float, float, float]:
        """Calculates pi_t: distribution over {low, mid, high} strengths."""
        t = self.global_step
        T = self.total_steps
        
        p_low = 1.0 - (t / T)
        p_high = t / T
        p_mid = (p_low + p_high) / 2.0
        
        total = p_low + p_mid + p_high
        return p_low / total, p_mid / total, p_high / total