
from pathlib import Path
from typing import List
import json

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from configSetup.configModel import ExperimentConfig
from configSetup.enums import MetricName


def plot_training(config: ExperimentConfig, run_dir: Path, budget: int):
    """
    Creates a training plot of a run tracking train/validation loss
    on one axis and the metric provided in the config as the monitor on the second loss.
    The plot is saved in the run dir.
    """

    csv_path = run_dir / f"metrics.csv"
    save_path = run_dir / f"training_plot.png"
    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="tab:blue", marker='o')
    ax1.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="tab:orange", marker='s')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df[config.training.monitor], color="tab:green", linestyle='--', label=f"Validation {config.training.monitor.value}")
    ax2.set_ylabel(f"Validation {config.training.monitor.value}")
    ax2.legend(loc="upper left")
    
    plt.title(f"Training/Validation Loss and Validation {config.training.monitor.value}\n{'_'.join(str(run_dir).split('/')[1:])}\nn={budget}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_budget_vs_metric(exp_root_dir: Path, budgets: List[int], metric: MetricName):
    """
    Plots mean metric vs budgets +- 1 standard deviation. Saves the plot in the experiment dir. 
    """
    means = []
    stds = []
    
    exp_dirs = [exp_root_dir / f"{budget}" for budget in budgets]
    
    for exp_dir in exp_dirs:
        json_path = exp_dir / "aggregated_best_metrics.json"
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found. Skipping {exp_dir}.")
            continue
            
        with open(json_path, "r") as f:
            data = json.load(f)
        try:
            mean_val = data[metric.value]["value"]["mean"]
            std_val = data[metric.value]["value"]["std"]
            means.append(mean_val)
            stds.append(std_val)
        except KeyError:
            print(f"Warning: Metric '{metric.value}' not found in {json_path}. Skipping.")
            
    np_budgets = np.array(budgets)
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(np_budgets, means, marker='o', color='tab:purple', label=f'Mean Best {metric.value}')
    
    ax.fill_between(np_budgets, means - stds, means + stds, color='tab:purple', alpha=0.2)
    
    ax.set_xlabel("Budget")
    ax.set_ylabel(f"Best {metric.value}")
    ax.set_title(f"Impact of Budget on Validation {metric.value}")
    ax.grid(True, linestyle='--', alpha=0.5)
        
    ax.legend()
    
    save_path = Path(exp_root_dir) / f"budgets_vs_{metric.value}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot successfully saved to {save_path}")


def plot_training_aggregates(config: ExperimentConfig, budget_dir: Path, budget: int):
    """
    Plots the mean training over n runs for a budget. 
    Saves the plot in the budget dir.
    """

    csv_path = budget_dir / "aggregated_metrics.csv"
    save_path = budget_dir / "aggregated_training_plot.png"
    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(df["epoch"], df["train_loss_mean"], label="Mean Train Loss", color="tab:blue", marker='o')
    ax1.fill_between(df["epoch"], 
                     df["train_loss_mean"] - df["train_loss_std"], 
                     df["train_loss_mean"] + df["train_loss_std"], 
                     color="tab:blue", alpha=0.2)
    
    ax1.plot(df["epoch"], df["val_loss_mean"], label="Mean Validation Loss", color="tab:orange", marker='s')
    ax1.fill_between(df["epoch"], 
                     df["val_loss_mean"] - df["val_loss_std"], 
                     df["val_loss_mean"] + df["val_loss_std"], 
                     color="tab:orange", alpha=0.2)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    monitor = config.training.monitor.value
    
    ax2.plot(df["epoch"], df[f'{monitor}_mean'], color="tab:green", linestyle='--', label=f"Mean Validation {monitor}")
    ax2.fill_between(df["epoch"], 
                     df[f'{monitor}_mean'] - df[f'{monitor}_std'], 
                     df[f'{monitor}_mean'] + df[f'{monitor}_std'], 
                     color="tab:green", alpha=0.2)
    ax2.set_ylabel(f"Mean {monitor}")
    ax2.legend(loc="upper left")
    

    plt.title(f"Mean Training/Validation Loss and Validation {monitor}\n{'_'.join(str(budget_dir).split('/')[1:])}\nn={budget} ({config.training.num_reps} reps)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)