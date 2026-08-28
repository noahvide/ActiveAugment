import ast
from glob import glob
import json
from pathlib import Path
import random

from matplotlib import patches
import pandas as pd

from activeLearning.agentStrategy import STRATEGY_COLORS
from configSetup.configEnums import TrainingType
from configSetup.configModel import load_config
from configSetup.enums import MetricName
from dataAugementation.intensity import AUGMENTATION_COLORS, get_intensity_shade

def parse_strategy(name):
    """Returns (base_name, is_adv)"""
    if str(name).endswith('_adv'):
        return name.replace('_adv', ''), True
    return name, False

def make_aug_int_label(r):
    aug, int_val = r['augmentation'], r['intensity']
    if aug in ["N/A", "DoNothing", "Flip"]: return aug
    if int_val in ['ALL', 'NONE', 'N/A', None]: return aug
    return f"{aug} ({int_val[0]})"

def sort_key(col_name):
    aug = col_name.split(' (')[0]
    int_order = {'(L)': 0, '(M)': 1, '(H)': 2}
    is_flip = 0 if aug == 'Flip' else 1
    return (is_flip, aug, int_order.get(col_name[-3:], 3))


def load_and_flatten_agent_data(budget_dir: Path) -> pd.DataFrame:
    all_records = []
    
    csv_files = glob(str(budget_dir / "run_*" / "agent_history.csv"))
    
    for file_path in csv_files:
        run_id = Path(file_path).parent.name  
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            step = row.get('step', 0)
            
            try:
                augs_batch = ast.literal_eval(str(row['selected_augmentations']))
                ints_batch = ast.literal_eval(str(row['selected_intensities']))
            except (ValueError, SyntaxError):
                continue
                
            for pipeline_idx, (pipeline_augs, pipeline_ints) in enumerate(zip(augs_batch, ints_batch)):
                for stage_idx, (aug, intensity) in enumerate(zip(pipeline_augs, pipeline_ints)):
                    all_records.append({
                        'run': run_id,
                        'step': step,
                        'batch_idx': pipeline_idx,
                        'stage': f"Stage {stage_idx + 1}",
                        'augmentation': aug,
                        'intensity': intensity
                    })           
    return pd.DataFrame(all_records)


def load_multibudget_agent_data(exp_root_dir: Path) -> pd.DataFrame:
    all_records = []
    budget_dirs = [d for d in exp_root_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for budget_dir in sorted(budget_dirs, key=lambda x: int(x.name)):
        budget_val = int(budget_dir.name)
        
        for file_path in budget_dir.glob("run_*/agent_history.csv"):
            run_id = file_path.parent.name
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    step = row.get('step', 0)
                    try:
                        augs_batch = ast.literal_eval(str(row['selected_augmentations']))
                        ints_batch = ast.literal_eval(str(row['selected_intensities']))
                    except (ValueError, SyntaxError):
                        continue
                        
                    for pipeline_idx, (pipeline_augs, pipeline_ints) in enumerate(zip(augs_batch, ints_batch)):
                        for stage_idx, (aug, intensity) in enumerate(zip(pipeline_augs, pipeline_ints)):
                            all_records.append({
                                'budget': budget_val,
                                'run': run_id,
                                'step': step,
                                'pipeline_idx': pipeline_idx,
                                'stage': f"Stage {stage_idx + 1}",
                                'augmentation': aug,
                                'intensity': intensity
                            })
            except Exception:
                continue
    return pd.DataFrame(all_records)

def load_multibudget_val_scores(exp_root_dir: Path) -> pd.DataFrame:
    budget_scores = []
    budget_dirs = [d for d in exp_root_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for budget_dir in budget_dirs:
        budget_val = int(budget_dir.name)
        
        for best_metrics_file in budget_dir.glob("run_*/best_metrics.json"):
            try:
                with open(best_metrics_file, 'r') as f:
                    best_data = json.load(f)
                
                f1_key = MetricName.F1_SCORE.value
                val_loss_key = MetricName.VAL_LOSS.value
                
                if f1_key in best_data:
                    final_score = best_data[f1_key]["value"]
                elif val_loss_key in best_data:
                    final_score = best_data[val_loss_key]["value"]
                else:
                    final_score = 0.0
                    
                budget_scores.append({
                    'budget': budget_val, 
                    'run': best_metrics_file.parent.name, 
                    'score': final_score
                })
            except Exception:
                continue
                
    return pd.DataFrame(budget_scores)

def load_multibudget_test_scores(exp_root_dir: Path) -> pd.DataFrame:
    test_records = []
    budget_dirs = [d for d in exp_root_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    for budget_dir in budget_dirs:
        budget_val = int(budget_dir.name)
        test_files = list(budget_dir.glob("aggregated_test_scores_*.json"))
        if test_files:
            try:
                with open(test_files[0], 'r') as f:
                    test_data = json.load(f)
                    metric_key = 'f1' if 'f1' in test_data['overall'] else list(test_data['overall'].keys())[0]
                    test_records.append({
                        'budget': budget_val,
                        'metric_name': metric_key.upper(),
                        'overall_mean': test_data['overall'][metric_key]['mean'],
                        'overall_std': test_data['overall'][metric_key]['std'],
                        'clean_mean': test_data['by_augmentation'].get("DoNothing", {}).get(metric_key, {}).get('mean', 0.0),
                        'clean_std': test_data['by_augmentation'].get("DoNothing", {}).get(metric_key, {}).get('std', 0.0)
                    })
            except Exception:
                continue
    return pd.DataFrame(test_records).sort_values('budget') if test_records else pd.DataFrame()


def load_multibudget_agent_scores(exp_root_dir: Path) -> pd.DataFrame:
    all_dfs = []
    budget_dirs = [d for d in exp_root_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for budget_dir in budget_dirs:
        budget_val = int(budget_dir.name)
        
        for agent_scores_file in budget_dir.glob("run_*/agent_scores.csv"):
            try:
                df = pd.read_csv(agent_scores_file)
                
                df['budget'] = budget_val
                df['run'] = agent_scores_file.parent.name
                
                all_dfs.append(df)
                
            except Exception as e:
                print(f"Failed to load {agent_scores_file}: {e}")
                continue
                
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def load_experiment_comparison_data(root_dir: Path):
    """Parses multiple strategies across multiple budgets."""
    metric_records, selection_records, test_records, test_aug_records, length_records, agent_scores_dfs = [], [], [], [], [], []
    config_file = None
    
    strategies = [d for d in root_dir.iterdir() if d.is_dir()]
    
    for strategy_path in strategies:
        strategy_name = strategy_path.name
        strategy_name = strategy_name.replace('_adv', '')
        
        trainer_type = TrainingType.STANDARD.value
        strat_config_file = strategy_path / "config_archive.yaml"
        if strat_config_file.exists():
            try:
                cfg = load_config(str(strat_config_file))
                trainer_type = cfg.training.type.value
            except Exception:
                pass
            
        if config_file is None:
            config_file = strat_config_file
            
        budget_dirs = [d for d in strategy_path.iterdir() if d.is_dir() and d.name.isdigit()]
        
        for budget_dir in budget_dirs:
            budget_val = int(budget_dir.name)
            
            test_file = budget_dir / "aggregated_test_scores_best_f1_score_model.json"
            if test_file.exists():
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                    
                    test_records.append({
                        'strategy': strategy_name, 
                        'trainer_type': trainer_type,
                        'budget': budget_val,
                        'test_f1_overall_mean': test_data['overall']['f1']['mean'],
                        'test_f1_overall_std': test_data['overall']['f1']['std'],
                        'test_f1_clean_mean': test_data['by_augmentation'].get("DoNothing", {}).get('f1', {}).get('mean', 0.0),
                        'test_f1_clean_std': test_data['by_augmentation'].get("DoNothing", {}).get('f1', {}).get('std', 0.0)
                    })
                    
                    for aug_name, metrics in test_data.get('by_augmentation', {}).items():
                        test_aug_records.append({
                            'strategy': strategy_name,
                            'trainer_type': trainer_type,
                            'budget': budget_val,
                            'augmentation': aug_name,
                            'test_f1_mean': metrics.get('f1', {}).get('mean', 0.0),
                            'test_f1_std': metrics.get('f1', {}).get('std', 0.0)
                        })
            
            for run_dir in budget_dir.glob("run_*"):
                best_metrics_file = run_dir / "best_metrics.json"
                history_file = run_dir / "agent_history.csv"
                scores_file = run_dir / "agent_scores.csv"
                
                if best_metrics_file.exists():
                    with open(best_metrics_file, 'r') as f:
                        best_data = json.load(f)
                        
                    best_val_f1 = best_data[MetricName.F1_SCORE.value]["value"]
                    
                    metric_records.append({
                        'strategy': strategy_name,
                        'trainer_type': trainer_type, 
                        'budget': budget_val, 
                        'val_f1': best_val_f1, 
                        'run': run_dir.name
                    })

                if scores_file.exists():
                    try:
                        df_score = pd.read_csv(scores_file)
                        df_score['strategy'] = strategy_name
                        df_score['trainer_type'] = trainer_type
                        df_score['budget'] = budget_val
                        df_score['run'] = run_dir.name
                        agent_scores_dfs.append(df_score)
                    except Exception as e:
                        print(f"Failed to load {scores_file}: {e}")
                        pass
                
                if history_file.exists():
                    h_df = pd.read_csv(history_file)
                    for _, row in h_df.iterrows():
                        aug_val = str(row.get('selected_augmentations', '')).strip()
                        if aug_val.lower() in ['nan', 'none', '[]', '']: continue
                            
                        try:
                            augs_batch = ast.literal_eval(aug_val)
                            for pipeline in augs_batch:
                                if isinstance(pipeline, list):
                                    active_len = sum(1 for a in pipeline if a not in ["DoNothing", "N/A"])
                                    length_records.append({'strategy': strategy_name, 'trainer_type': trainer_type, 'budget': budget_val, 'length': active_len})
                                    
                                    for aug in pipeline:
                                        if aug not in ["DoNothing", "N/A"]:
                                            selection_records.append({'strategy': strategy_name, 'trainer_type': trainer_type, 'augmentation': aug})
                                else:
                                    if pipeline not in ["DoNothing", "N/A"]:
                                        selection_records.append({'strategy': strategy_name, 'trainer_type': trainer_type, 'augmentation': pipeline})
                                    length_records.append({'strategy': strategy_name, 'trainer_type': trainer_type, 'budget': budget_val, 'length': 0 if pipeline in ["DoNothing", "N/A"] else 1})
                        except (ValueError, SyntaxError):
                            continue

    df_scores = pd.concat(agent_scores_dfs, ignore_index=True) if agent_scores_dfs else pd.DataFrame()

    return (pd.DataFrame(metric_records), 
            pd.DataFrame(test_records), 
            pd.DataFrame(test_aug_records),
            pd.DataFrame(selection_records), 
            pd.DataFrame(length_records), 
            df_scores,
            config_file)

def add_master_strategy_legend(fig, ax, title="Strategy"):
    """Extracts legend handles from a specific axis and creates a unified global legend."""
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend(): ax.get_legend().remove()
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=max(1, len(labels)//2), title=title)


def add_master_augmentation_legend(fig, df_active, palette, shade_func, sort_key):
    """Builds the Augmentation (Intensity) legend from raw data. Use for Agent/Budget dashboards."""
    active_ordered_cols = sorted(list(set(df_active['aug_with_intensity'])), key=sort_key, reverse=True)
    intensity_map = {'H': 'HIGH', 'M': 'MEDIUM', 'L': 'LOW', 'A': 'ALL'}
    
    aug_groups = {}
    for col in active_ordered_cols:
        aug = col.split(' (')[0]
        if aug not in aug_groups:
            aug_groups[aug] = []
        aug_groups[aug].append(col)
        
    columns = list(aug_groups.values())
    dynamic_ncol = max(1, len(columns))
    
    max_rows = max([len(col) for col in columns]) if columns else 0
    
    legend_handles = []
    
    for col_list in columns:
        for row_idx in range(max_rows):
            if row_idx < len(col_list):
                col_name = col_list[row_idx]
                parts = col_name.split(' (')
                aug = parts[0]
                int_short = parts[1][0] if len(parts) > 1 else 'A'
                
                shaded_color = shade_func(palette.get(aug, '#000000'), intensity_map.get(int_short, 'ALL'))
                legend_handles.append(patches.Patch(color=shaded_color, label=col_name))
            else:
                legend_handles.append(patches.Patch(color='none', label=''))

    fig.legend(
        handles=legend_handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.02), 
        ncol=dynamic_ncol,
        title="Active Augmentation (Intensity)"
    )

def generate_dynamic_palette(strategies: list[str]) -> dict:
    palette = {}
    aug_color_lower = {k.lower(): v for k, v in AUGMENTATION_COLORS.items()}
    used_colors = set()
    
    CUSTOM_MAP = {
        "F_F_F_F": "#633123",
        "F_F_T_F": "#BA5F46",
        "F_F_T_T": "#E8BDB1",
        "T_F_F_F": "#4F5D69",
        "T_F_T_F": "#758B9B",
        "T_F_T_T": "#C8D3DB",
        "T_T_F_F": "#445028",
        "T_T_T_F": "#67783C",
        "T_T_T_T": "#BDC9A0",
}
    
    
    for display_name in strategies:
        base_strat = display_name.replace(" (adv)", "")
        
        suffix = "_".join(base_strat.split("_")[1:])
        
        if suffix in CUSTOM_MAP:
            color = CUSTOM_MAP[suffix]
        
        elif base_strat in STRATEGY_COLORS:
            color = STRATEGY_COLORS[base_strat]
        
        elif base_strat.split('_')[0].lower() in aug_color_lower:
            parts = base_strat.split('_')
            base_color = aug_color_lower[parts[0].lower()]
            intensity = parts[1].upper() if len(parts) > 1 else 'ALL'
            color = get_intensity_shade(base_color, intensity)
            
        else:
            while True:
                r = random.randint(20, 220)
                g = random.randint(20, 220)
                b = random.randint(20, 220)
                color = f"#{r:02x}{g:02x}{b:02x}".lower()
                
                if color not in used_colors:
                    break 
            
        palette[display_name] = color
        used_colors.add(color.lower())
                
    return palette
    
# def generate_dynamic_palette(strategies: list[str]) -> dict:
#     """Builds a complete color palette for any mix of standard and augmentation-based strategies."""
#     palette = {}
    
#     aug_color_lower = {k.lower(): v for k, v in AUGMENTATION_COLORS.items()}
    
#     used_colors = set()
    
#     for strat in strategies:
#         if strat in STRATEGY_COLORS:
#             color = STRATEGY_COLORS[strat]
#             palette[strat] = color
#             used_colors.add(color.lower())
#             continue
            
#         parts = strat.split('_')
#         base_name = parts[0].lower()
        
#         if base_name in aug_color_lower:
#             base_color = aug_color_lower[base_name]
            
#             intensity = parts[1].upper() if len(parts) > 1 else 'ALL'
#             if intensity not in ['LOW', 'MEDIUM', 'HIGH', 'ALL']:
#                 intensity = 'ALL'
                
#             color = get_intensity_shade(base_color, intensity)
#             palette[strat] = color
#             used_colors.add(color.lower())
            
#         else:
#             while True:
#                 r = random.randint(20, 220)
#                 g = random.randint(20, 220)
#                 b = random.randint(20, 220)
#                 random_color = f"#{r:02x}{g:02x}{b:02x}".lower()
                
#                 if random_color not in used_colors:
#                     palette[strat] = random_color
#                     used_colors.add(random_color)
#                     break 
                    
#     return palette

def load_single_budget_agent_scores(budget_dir: Path) -> pd.DataFrame:
    """Loads the agent confidence scores for all runs within a single budget."""
    all_dfs = []
    for run_dir in budget_dir.glob("run_*"):
        scores_file = run_dir / "agent_scores.csv"
        if scores_file.exists():
            try:
                df = pd.read_csv(scores_file)
                df['run'] = run_dir.name
                all_dfs.append(df)
            except Exception as e:
                print(f"Failed to load {scores_file}: {e}")
                
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def prepare_augmentation_data(df_augs: pd.DataFrame) -> pd.DataFrame:
    df_plot = df_augs.copy()
    
    df_plot['base_augmentation'] = df_plot['augmentation'].apply(
        lambda x: str(x).split('_')[0] if x not in ['DoNothing', 'N/A'] else str(x)
    )
    
    df_agg = df_plot.groupby(['strategy', 'budget', 'base_augmentation'], as_index=False).agg({
        'test_f1_mean': 'mean',
        'test_f1_std': 'mean' 
    })
    
    return df_agg