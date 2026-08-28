from datetime import datetime
from pathlib import Path
import platform
from typing import Optional, Tuple

import torch
import yaml

from configSetup.configModel import ExperimentConfig, load_config, DatasetSelectionType, AugmentationStrategyType
from dataAugementation.augmentationFactory import augmentation_factory
from dataProcessing.datasets.validation_dataset import ValidationDataset
from dataProcessing.statsRegistry import get_dataset_stats
from models.modelFactory import get_model
from training.utils import get_classes, get_git_revision_hash, get_model_summary

def parseConfig(config_pth) -> Tuple[Optional[ExperimentConfig], Path]:
    config: ExperimentConfig = load_config(config_pth)

    exp_name_str = config.experiment_name
    data_root = config.dataset.name
    model_str = config.model.name  
    weights_status = config.model.weights_status
    config_name = config.config_name
    
    base_path: Path = Path("experiments") / exp_name_str / data_root.name / model_str.value / weights_status.value / config_name

    base_path.mkdir(parents=True, exist_ok=True)
    
    with open(base_path / "config_archive.yaml", "w") as f:
        yaml.dump(config.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False) 
        
    budgets = []
    if config.dataset_selection.type == DatasetSelectionType.ACTIVE and config.dataset_selection.budgets:
        budgets = config.dataset_selection.budgets
        for budget in budgets:
            (base_path / str(budget)).mkdir(parents=True, exist_ok=True)
    else:
        (base_path / "full").mkdir(parents=True, exist_ok=True)

    aug_strings = []
    if config.augmentation_strategy.type != AugmentationStrategyType.NONE and config.augmentation_space:
        config_augs = config.augmentation_space.augmentations
        augmentations_list = augmentation_factory(config_augs, config.augmentation_space.fixed_pipeline_length, config.dataset.name)
        aug_strings = [f"{str(aug)} {aug.get_range()}" for aug in augmentations_list]

    val_ds = ValidationDataset(root_dir=data_root, weights_status=config.model.weights_status)

    model_name = config.model.name
    classes = get_classes(data_root)
    model = get_model(model_name, len(classes), config.model.weights_status)
    
    model_summary = get_model_summary(model)
    model_summary["weights_status"] = weights_status.value
    model_summary["model_name"] = model_name.value
    
    stats = get_dataset_stats(data_root)
    mean = stats["mean"]
    std = stats["std"]
    
    training_setup = config.training.model_dump(mode='json')
    training_setup["optimizer"] = "Adam"

    metadata = {
        "experiment": {
            "name": config.experiment_name,
            "config_name": config.config_name,
            "dataset_selection": config.dataset_selection.type.value,
            "augmentation_strategy": config.augmentation_strategy.type.value,
            "root_dir": str(base_path),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_hash": get_git_revision_hash(),
        },
        "hardware": {
            "device": config.training.device.value if hasattr(config.training.device, 'value') else str(config.training.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "python_version": platform.python_version(),
            "pytorch_version": str(torch.__version__)
        },
        "datasets": {
            "name": data_root.name,
            "normalization_mean": mean,
            "normalization_std": std,
            "budgets": budgets if budgets else "FULL_DATASET",
            "val_samples": len(val_ds),
            "classes": classes,
            "img_size": getattr(config.dataset, 'img_size', 256), 
            "seed": config.dataset.seed,
            "num_reps": config.training.num_reps
        },
        "model_summary": model_summary,
        "training": training_setup,
        
        "dataset_selection_details": config.dataset_selection.model_dump(mode='json'),
        "augmentation_strategy_details": config.augmentation_strategy.model_dump(mode='json')
    }
    
    if config.augmentation_strategy.type != AugmentationStrategyType.NONE and config.augmentation_space:
        metadata["augmentations_space"] = {
            "pipeline_length": config.augmentation_space.pipeline_length,
            "fixed_pipeline_length": config.augmentation_space.fixed_pipeline_length,
            "space": aug_strings
        }
        
    with open(base_path / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    return config, base_path