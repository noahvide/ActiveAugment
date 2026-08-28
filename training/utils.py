import json
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from torchvision import datasets

from configSetup.configModel import ExperimentConfig
from dataProcessing.dataPath import DataPath
from models.modelFactory import ALModel




def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Not a git repository"

def get_model_summary(model: ALModel):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "model_name": model.__class__.__name__
    }

def get_classes(data_root: DataPath):
    return datasets.ImageFolder(root=str(Path(data_root.get_path) / "Train")).classes


def aggregate_json_files(config: ExperimentConfig, budget_dir: Path, filename: str):
    """
    Reads a specific JSON file from all runs, aggregates values, and computes mean/std.
    Saves scores to a json file in budget dir.
    """
    
    accumulated_data = {}
    def accumulate(source, target):
        for key, value in source.items():
            if isinstance(value, dict):
                if key not in target:
                    target[key] = {}
                accumulate(value, target[key])
            else:
                if key not in target:
                    target[key] = []
                target[key].append(value)

    for run_dir in budget_dir.glob("run_*"):
        file_path = run_dir / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                accumulate(data, accumulated_data)
        else:
            print(f"Warning: {file_path} not found.")

    def compute_stats(target):
        stats = {}
        for key, value in target.items():
            if isinstance(value, dict):
                stats[key] = compute_stats(value)
            else:
                stats[key] = {
                    "mean": float(np.mean(value)),
                    "std": float(np.std(value))
                }
        return stats

    final_stats = compute_stats(accumulated_data)
    
    out_path = budget_dir / f"aggregated_{filename}"
    with open(out_path, "w") as f:
        json.dump(final_stats, f, indent=4)
        
    print(f"Aggregated {filename} saved to {out_path}")

def aggregate_csv_metrics(config: ExperimentConfig, budget_dir: Path):
    """
    Reads CSV metrics from all runs, groups by epoch, and calculates mean/std for numbers, and joins strings.
    Saves scores to a csv file in budget dir.
    """
    
    dfs = []
    
    for run_dir in budget_dir.glob("run_*"):
        file_path = run_dir / "metrics.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found.")
            
    if not dfs:
        print(f"No metrics.csv files found to aggregate.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    
    numeric_cols = combined_df.select_dtypes(include=['number']).columns.drop('epoch', errors='ignore')
    string_cols = combined_df.select_dtypes(exclude=['number']).columns
    
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'std']
        
    for col in string_cols:
        agg_dict[col] = [lambda x: str(x.dropna().astype(str).unique().tolist())]
    
    aggregated_df = combined_df.groupby('epoch').agg(agg_dict)
    
    flat_columns = []
    for col in aggregated_df.columns:
        if col[1] == '<lambda>':
            flat_columns.append(f"{col[0]}_unique")
        else:
            flat_columns.append(f"{col[0]}_{col[1]}")
            
    aggregated_df.columns = flat_columns
    aggregated_df = aggregated_df.reset_index()
    
    out_path = budget_dir / f"aggregated_metrics.csv"
    aggregated_df.to_csv(out_path, index=False)
    
    print(f"Aggregated metrics.csv saved to {out_path}")
