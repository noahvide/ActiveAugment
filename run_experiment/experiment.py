from datetime import datetime
import os
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import seed_everything

from configSetup.configModel import AugmentationStrategyType, DatasetSelectionType, ExperimentConfig, TrainingType

from activeLearning.agentStrategy import AgentStrategy
from activeLearning.agent_factory import get_agent 

from dataAugementation.augmentationService import AugmentationService
from dataAugementation.augmentationFactory import augmentation_factory

from dataProcessing.datasets import CachedDataset, ValidationDataset, TrainingDataset
from dataProcessing.datasetService import DatasetService

from models.modelFactory import get_model

from training.trainers import ActiveAugmentationTrainer, StandardTrainer, ActiveInvarianceTrainer, MaxUpTrainer
from plotting.trainingPlotting import plot_training_aggregates
from training.utils import aggregate_csv_metrics, aggregate_json_files


def run_config(config: ExperimentConfig, exp_root_dir: Path):
    start_time = time.time()
    print(f"Running Config: {config.config_name}")
    print(f" - Dataset Strategy: {config.dataset_selection.type.value} ({config.dataset_selection.strategy.value if config.dataset_selection.strategy else 'N/A'})")
    print(f" - Augmentation Strategy: {config.augmentation_strategy.type.value} ({config.augmentation_strategy.agent.value if config.augmentation_strategy.agent else 'N/A'})")
    print(f" - Trainer: {config.training.type.value}")
    
    val_ds = ValidationDataset(root_dir=config.dataset.name, weights_status=config.model.weights_status)
    val_loader = DataLoader(CachedDataset(val_ds), batch_size=config.training.batch_size, num_workers=0, shuffle=False)
    num_classes = len(val_ds.classes)
    
    
    seeds = []
    budgets = []
    
    for i in range(1, config.training.num_reps + 1):
        current_seed = config.dataset.seed + i 
        seeds.append(current_seed)
        seed_everything(current_seed)
        
        print(f"--- Starting Repetition {i}/{config.training.num_reps} ---")
        if config.augmentation_space:
            augmentation_space = augmentation_factory(config.augmentation_space.augmentations, 
                                                      config.augmentation_space.fixed_pipeline_length, 
                                                      config.dataset.name)
        else:
            augmentation_space = None
        aug_service = AugmentationService(augmentation_space)
        
        full_train_ds = TrainingDataset(root_dir=config.dataset.name, weights_status=config.model.weights_status)
        dataset_manager = DatasetService(full_train_ds, device=config.training.device)
        aug_service.set_normalizer(full_train_ds.norm_transform)
        
        
        budgets = config.dataset_selection.budgets if config.dataset_selection.strategy != DatasetSelectionType.FULL and config.dataset_selection.budgets else [len(full_train_ds)]
        
        model = get_model(config.model.name, num_classes, config.model.weights_status)
        dataset_agent = get_agent(config, config.dataset_selection.strategy, augmentation_space, current_seed, feature_dim=model.feature_dim)

        for step, budget in enumerate(sorted(budgets)):
            run_dir = exp_root_dir / str(budget) / f"run_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)
                        
            if config.dataset_selection.type == DatasetSelectionType.FULL:
                dataset_manager.acquire_random_stratified(budget, current_seed)
            elif config.dataset_selection.type == DatasetSelectionType.ACTIVE:
                if step == 0:
                    dataset_manager.acquire_random_stratified(budget, current_seed)
                else:
                    num_to_acquire = budget - dataset_manager.current_labeled_size
                    print(f"Acquiring {num_to_acquire} images using {config.dataset_selection.strategy.value if config.dataset_selection.strategy else 'N/A'} strategy...")
                    
                    if config.dataset_selection.strategy == AgentStrategy.RANDOM:
                        dataset_manager.acquire_random_stratified(num_to_acquire, current_seed)
                    else:
                        dataset_manager.acquire_labels(model, dataset_agent, num_to_acquire)
                                        
                    if not config.dataset_selection.continual_learning:
                        del model
                        import gc; gc.collect(); torch.cuda.empty_cache()
                        model = get_model(config.model.name, num_classes, config.model.weights_status)
                unlabeled_loader = dataset_manager.get_unlabeled_loader(batch_size=config.training.batch_size)
            else:
                raise NotImplementedError(f"data selection type {config.dataset_selection.type} not implemented")                

            train_loader = dataset_manager.get_train_loader(batch_size=config.training.batch_size)
            

            aug_type = config.augmentation_strategy.type

            # Determine Augmentation Selection Strategy
            if aug_type == AugmentationStrategyType.ACTIVE:
                # ActiveAugment
                aug_agent = get_agent(config, config.augmentation_strategy.agent, augmentation_space, current_seed, feature_dim=model.feature_dim)
            elif aug_type == AugmentationStrategyType.STATIC:
                # Apply pipeline
                aug_agent = None
            elif aug_type == AugmentationStrategyType.NONE:
                # No Augmentation
                aug_agent = None
            else:
                raise NotImplementedError(f"Unknown augmentation strategy type: {aug_type}")

            
            # Determine Training Strategy (loss)
            if config.training.type == TrainingType.CONTRASTIVE:
                trainer = ActiveInvarianceTrainer(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    config=config, output_dir=run_dir, run=i, agent=aug_agent,
                    aug_service=aug_service, budget=budget
                )
            elif config.training.type == TrainingType.MAXUP:
                trainer = MaxUpTrainer(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    config=config, output_dir=run_dir, run=i, budget=budget, 
                    aug_service=aug_service
                )
            elif config.training.type == TrainingType.STANDARD:
                if aug_type in [AugmentationStrategyType.ACTIVE, AugmentationStrategyType.STATIC]:
                    trainer = ActiveAugmentationTrainer(
                        model=model, train_loader=train_loader, val_loader=val_loader,
                        config=config, output_dir=run_dir, run=i, agent=aug_agent,
                        budget=budget, aug_service=aug_service
                    )
                else:
                    trainer = StandardTrainer(
                        model=model, train_loader=train_loader, val_loader=val_loader,
                        config=config, output_dir=run_dir, run=i, budget=budget
                    )
            else:
                raise ValueError(f"Unknown training type: {config.training.type}")
            
            trainer.fit()

            if aug_type == AugmentationStrategyType.ACTIVE and aug_agent:
                if aug_agent.aug_scores:
                    pd.DataFrame(aug_agent.aug_scores).to_csv(run_dir / "agent_scores.csv", index=False)
                
                if aug_agent.history:
                    pd.DataFrame(aug_agent.history).to_csv(run_dir / "agent_history.csv", index=False)

            if config.training.do_evaluation:
                trainer.evaluate()

            # Optional cleanup of checkpoint files after training and evaluation to save disk space.              
            # if trainer.checkpoint_path and os.path.exists(trainer.checkpoint_path):
            #     os.remove(trainer.checkpoint_path)

    for budget in budgets:
        budget_dir = exp_root_dir / str(budget)
        with open(budget_dir / "metadata.yaml", "a") as f:
            f.write(f"\nDatetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            f.write(f"\nNumber of runs: {config.training.num_reps}")
            f.write(f"\nSeeds used for each run: {seeds}")
        
        aggregate_json_files(config, budget_dir, "best_metrics.json")
        aggregate_csv_metrics(config, budget_dir)
        if config.training.do_evaluation:
            aggregate_json_files(config, budget_dir, "test_scores_best_f1_score_model.json")
        plot_training_aggregates(config, budget_dir, budget)
        
    duration = time.time() - start_time
    print(f"Experiment completed in {duration/60:.2f} minutes.")