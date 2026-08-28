import argparse
from pathlib import Path

from plotting.strategyPlotting import generate_experiment_comparison, generate_strategy_dashboard, plot_augmentation_facet_grid
from run_experiment.parseConfig import parseConfig
from run_experiment.experiment import run_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a yaml config or a folder of configs")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if config_path.is_dir():
        config_files = sorted(list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml")))
        print(f"Found {len(config_files)} configurations in directory: {args.config}")
    else:
        config_files = [config_path]

    for pth in config_files:        
        exp_config, exp_root_dir = parseConfig(config_pth=str(pth))
        
        if exp_config:
            run_config(exp_config, exp_root_dir)            
            root_dir = exp_root_dir.parent
            
            generate_strategy_dashboard(exp_config, exp_root_dir)
            plot_augmentation_facet_grid(root_dir)
            generate_experiment_comparison(root_dir)
            
