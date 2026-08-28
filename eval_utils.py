import json
import os
import yaml

from glob import glob
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Literal, Optional

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision

from configSetup.enums import MetricName
from dataAugementation.augmentationService import AugmentationService
from dataAugementation.augmentations import AUGMENTATION_MAP, AugmentationEnum
from dataAugementation.intensity import INTENSITY_MAP, Intensity

from dataProcessing.dataPath import DataPath
from dataProcessing.datasets.cachedDataset import CachedDataset
from dataProcessing.datasets.test_dataset import TestDataset

from models.modelEnums import ModelArchitecture, WeightsStatus
from models.modelFactory import ALModel, get_model

from utils import CURRENT_SEED, seed_everything



def load_checkpoint(exp_dir, map_location="cpu", strict=True):
    checkpoint_path = f"{exp_dir}/best_model.pth"
    with open(f"{exp_dir}/metadata.yaml", "r") as file:
        meta_data = yaml.safe_load(file)
        
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    model_name = meta_data["model_summary"]["model_name"]
    if model_name == "ResNet":
        model_name = ModelArchitecture.RESNET18
    elif model_name == "TinyVit":
        model_name = ModelArchitecture.TINYVIT
    model = get_model(model_name, len(meta_data["dataset"]["classes"]), weights_status=WeightsStatus.RANDOM)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=strict)

    if missing_keys:
        raise Exception(f"The following keys are missing in checkpoint: {missing_keys}")
    if unexpected_keys:
        raise Exception(f"The following unexpected keys are present in checkpoint: {unexpected_keys}")

    return model



def rename(dataset: DataPath, model: ModelArchitecture, pretrained: bool):
    dataset_name = dataset.name
    
    if pretrained:
        exp_path = f"./experiments_results/{dataset_name}/{model}/pretrained/*"
    else:
        exp_path = f"./experiments_results/{dataset_name}/{model}/scratch/*"
    
    for path in glob(exp_path):
        folder_name = path.split("/")[-1]
        new_folder_name = str.join("_", folder_name.split("_")[3:])
        
        parent_dir = os.path.dirname(path)
        new_path = os.path.join(parent_dir, new_folder_name)
        
        print(f"Renaming: {folder_name} -> {new_folder_name}")
        
        try:
            os.rename(path, new_path)
        except OSError as e:
            print(f"Error renaming {folder_name}: {e}")


def get_test_suite() -> Dict:
    test_suite = {"DoNothing": []}
    
    for aug_name, aug_class in AUGMENTATION_MAP.items():
        if aug_name not in [AugmentationEnum.FLIP, AugmentationEnum.CONTRAST, AugmentationEnum.BRIGHTNESS, AugmentationEnum.GAUSSIANBLUR, AugmentationEnum.GAUSSIANNOISE, AugmentationEnum.ROTATION]:
            continue
        if aug_name == AugmentationEnum.FLIP:
            variant_name = f"{aug_name.name}"
            test_suite[variant_name] = [aug_class()] # type: ignore
        else:    
            for _, intensity in INTENSITY_MAP.items():
                if intensity != Intensity.ALL:
                    variant_name = f"{aug_name.name}_{intensity.name}"
                    test_suite[variant_name] = [aug_class(intensity)]
    return test_suite

@torch.no_grad()
def get_test_scores(model: ALModel, checkpoint_path: Path, test_ds: TestDataset, device: torch.device, test_seed: int = 0):
    """
    Evaluates a model on a test set containing all possible augmentations. 
    """
    current_seed = CURRENT_SEED
    seed_everything(test_seed)
    cached_test = CachedDataset(test_ds)
    test_loader = DataLoader(cached_test, batch_size=32, num_workers=0, shuffle=False)
    
    num_classes = len(test_ds.classes) 
    
    def create_metrics():
        return MetricCollection({
            'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro'),
            'acc': Accuracy(task="multiclass", num_classes=num_classes),
            'prec': Precision(task="multiclass", num_classes=num_classes, average='macro'),
            'class_acc': Accuracy(task="multiclass", num_classes=num_classes, average='none'),
            'class_f1': F1Score(task="multiclass", num_classes=num_classes, average='none'),
            'class_prec': Precision(task="multiclass", num_classes=num_classes, average='none')
        }).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    _ = model.load_state_dict(checkpoint, strict=True)
    
    model.eval()
    
    test_suite = get_test_suite()
    
    overall_metrics = create_metrics()
    aug_metrics = {variant: create_metrics() for variant in test_suite.keys()}
    total_loss_dict = {variant: 0.0 for variant in test_suite.keys()}
    
    
    augmentation_parameters = {variant: [] for variant in test_suite.keys()}
    
    aug_service = AugmentationService(augmentations=None, norm_transform=test_loader.dataset.norm_transform) # type: ignore
        
    with torch.no_grad():
        for variant_name, aug_list in tqdm(test_suite.items(), desc="Getting test scores", leave=False):
            for images, labels_idx in test_loader:
                images, labels_idx = images.to(device), labels_idx.to(device)
                
                augmented_images, aug_info = aug_service.apply_pipeline(images, aug_list, 1.0)
                
                outputs = model(augmented_images)
                loss = criterion(outputs, labels_idx).item()
                
                total_loss_dict[variant_name] += loss
                aug_metrics[variant_name].update(outputs, labels_idx)
                overall_metrics.update(outputs, labels_idx)
                vals = []
                for img_candidates_meta in aug_info:
                    pipeline_meta = img_candidates_meta[0].get("pipeline", [])
                    
                    if pipeline_meta:
                        val = pipeline_meta[0].get("sampled_value", 0.0)
                        vals.append(val)
                    else:
                        vals.append(0.0)
                        
                augmentation_parameters[variant_name].extend(vals)
                
    results = {'overall': {}, 'by_augmentation': {}, 'by_class': {}, 'sampled_values': augmentation_parameters}
    
    computed_overall = overall_metrics.compute()
    results['overall'] = {
        'acc': computed_overall['acc'].item(),
        'f1': computed_overall['f1'].item(),
        'prec': computed_overall['prec'].item(),
        'test_loss': sum(total_loss_dict.values()) / (len(test_loader) * len(test_suite))
    }
    
    for i, class_name in enumerate(test_ds.classes):
        results['by_class'][class_name] = {
            'acc': computed_overall['class_acc'][i].item(),
            'f1': computed_overall['class_f1'][i].item(),
            'prec': computed_overall['class_prec'][i].item()
        }
    
    for variant, metrics in aug_metrics.items():
        computed = metrics.compute()
        results['by_augmentation'][variant] = {
            'acc': computed['acc'].item(),
            'f1': computed['f1'].item(),
            'prec': computed['prec'].item()
        }
    
    seed_everything(current_seed)
    return results

def plot_best_metrics(root_dir, dataset: DataPath, model: ModelArchitecture, weights_status: WeightsStatus, split: Literal["Val", "Test"], test_part=None, metric: MetricName = MetricName.F1_SCORE):
    results = {}
    dataset_name = dataset.name
    
    exp_path = f"./{root_dir}/{dataset_name}/{model}/{weights_status}/*"

    
    for path in glob(exp_path):
        folder_name = path.split("/")[-1]
        
        if folder_name.lower() in ["none", "baseline", "original"]:
            augmentation = "Baseline"
            intensity = "ORIGINAL"
        else:
            parts = folder_name.split("_")
            augmentation = parts[0]
            intensity = parts[-1]

        if augmentation not in results:
            results[augmentation] = {}

        if split == "Val":
            try:
                with open(f"{path}/best_metrics.json", mode="r") as file:
                    content = json.load(file)
                    if metric == "f1":
                        results[augmentation][intensity] = content["f1_score"]["value"]
                    elif metric == "loss":
                        results[augmentation][intensity] = content["val_loss"]["value"]
                    else:
                        raise ValueError("Metric must be either f1 or loss")
            except (FileNotFoundError, ValueError, IndexError):
                continue
        elif split == "Test":
            try:
                if metric == "f1":
                    score_path = f"{path}/test_scores_best_f1_score_model.json"
                elif metric == "loss":
                    score_path = f"{path}/test_scores_best_val_loss_model.json"
                else:
                    raise ValueError("Metric must be either f1 or loss")
                with open(score_path, mode="r") as file:
                    content = json.load(file)
                    if test_part:
                        if test_part.lower() == "original":
                            best_score = content["by_augmentation"]["N/A_N/A"]["f1"]
                        else:
                            best_score = content["by_augmentation"][test_part]["f1"]
                    else:
                        best_score = content["overall"]["f1"]
                    results[augmentation][intensity] = best_score
            except (FileNotFoundError, ValueError, IndexError) as e:
                continue
    
    df = pd.DataFrame(results).T
    
    baseline_value = None
    if "Baseline" in df.index:
        baseline_value = df.loc["Baseline", "ORIGINAL"]
        df = df.drop("Baseline")
    
    intensity_order = ['LOW', 'MEDIUM', 'HIGH']
    for level in intensity_order:
        if level not in df.columns:
            df[level] = np.nan
            
    df = df[intensity_order]
    print(df)
    
    fig, ax = plt.subplots(figsize=(14, 7), layout='constrained')
    width = 0.8
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    df.plot(kind='bar', ax=ax, width=width, color=colors, alpha=0.6)

    if baseline_value is not None:
        ax.axhline(y=baseline_value, color='black', linestyle='--', linewidth=1.5, label=f'No augmentations ({baseline_value:.4f})') # type: ignore

    n_bars = len(df.columns)
    bar_width = width / n_bars

    for i, (index, row) in enumerate(df.iterrows()):
        valid_data = row.dropna()
        if valid_data.empty:
            continue
            
        best_col = valid_data.idxmax()
        worst_col = valid_data.idxmin()
        
        positions = {col: i + (j - (n_bars - 1) / 2) * bar_width for j, col in enumerate(df.columns)}
        
        ax.annotate('Best', xy=(positions[best_col], row[best_col]), # type: ignore
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkgreen')
        
        if len(valid_data) > 1 and best_col != worst_col:
            ax.annotate('Worst', xy=(positions[worst_col], row[worst_col]), # type: ignore
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkred')

    if split == "Val":
        plt.title(f'Augmentation Impact vs Baseline: {model} ({weights_status} weights) on {dataset_name} (Validation set)', fontsize=14)
    elif split == "Test":
        if test_part:
            plt.title(f'Augmentation Impact vs Baseline: {model} ({weights_status} weights) on {dataset_name} ({"Original data" if test_part.lower() == "original" else test_part} of Test set)', fontsize=14)
        else:
            plt.title(f'Augmentation Impact vs Baseline: {model} ({weights_status} weights) on {dataset_name} (Test set)', fontsize=14)
            
    
    plt.xlabel('Augmentation Type', fontsize=12)
    plt.ylabel('f1-score', fontsize=12)
    plt.xticks(rotation=45)
    
    # max_val = max(np.nanmax(df.values), baseline_value if baseline_value else 0)
    plt.ylim(0, 1.1)
    
    plt.legend(title='Intensity / Reference', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
    


def plot_evaluation_results(root_dir, dataset: DataPath, model_name: ModelArchitecture, weights_status: Literal["frozen", "tunable", "random"], aug_str: str,  metric: Literal["f1", "loss"]="f1", save_path=None):
    dataset_name = dataset.name
    
    exp_path = f"./{root_dir}/{dataset_name}/{model_name}/{weights_status}/{aug_str}"

    
    if metric == "f1":
        score_path = f"{exp_path}/test_scores_best_f1_score_model.json"
    elif metric == "loss":
        score_path = f"{exp_path}/test_scores_best_val_loss_model.json"
    else:
        raise ValueError("Metric must be either f1 or loss")
    with open(score_path, mode="r") as file:
        results_dict = json.load(file)

    aug_data = defaultdict(dict)
    baseline_val = 0
    
    for key, metrics in results_dict.get('by_augmentation', {}).items():
        if key in ['N/A_N/A', 'None_None']:
            baseline_val = metrics.get(metric, 0)
            continue
        
        parts = key.rsplit('_', 1)
        aug_name, intensity = (parts[0], parts[1]) if len(parts) == 2 else (key, "UNKNOWN")
        aug_data[aug_name][intensity] = metrics.get(metric, 0)

    aug_names = sorted(list(aug_data.keys()))
    aug_names.append("Original")
    
    standard_intensities = ['LOW', 'MEDIUM', 'HIGH']
    found_intensities = set()
    for intensities in aug_data.values():
        found_intensities.update(intensities.keys())
        
    plot_intensities = [i for i in standard_intensities if i in found_intensities]
    plot_intensities += sorted([i for i in found_intensities if i not in standard_intensities])

    x = np.arange(len(aug_names))
    width = 0.8 / len(plot_intensities) 
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    base_color = '#95a5a6'
    
    for i, intensity in enumerate(plot_intensities):
        values = []
        bar_colors = [] 
        
        for name in aug_names:
            if name == "Original":
                values.append(baseline_val if i == 1 else 0)
                bar_colors.append(base_color)
            else:
                values.append(aug_data[name].get(intensity, 0))
                bar_colors.append(colors[i] if i < len(colors) else '#34495e')
        
        offset = (i - len(plot_intensities) / 2 + 0.5) * width
        
        rects = ax.bar(x + offset, values, width, label=intensity, 
                       color=bar_colors, edgecolor='white', alpha=0.6)
        
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                


    metric_label = metric.upper() if metric != 'acc' else 'Accuracy'
    
    if aug_str.lower() == "none":
        ax.set_title(f'{metric_label} Test Score for {model_name} ({dataset_name}) categorized by augmentation\nModel trained on original data with {weights_status} weights', 
                    fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(f'{metric_label} Test Score for {model_name} ({dataset_name}) categorized by augmentation\nModel trained with augmentation {aug_str} with {weights_status} weights', 
                    fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(f'{metric_label} Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(aug_names, rotation=45, ha='right', fontsize=11)
    
    ax.legend(title='Intensity', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_ylim(0, 1.1) 
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
        