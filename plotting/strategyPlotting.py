from pathlib import Path

from matplotlib import gridspec, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configSetup.configModel import ExperimentConfig, load_config
from dataAugementation.intensity import AUGMENTATION_COLORS, get_intensity_shade


import matplotlib.gridspec as gridspec
from pathlib import Path

from plotting.plot_modules import plot_agent_confidence, plot_comparison_confidence_heatmap, plot_comparison_heatmap, plot_comparison_length_heatmap, plot_comparison_test_metrics, plot_comparison_val_metrics, plot_global_active_frequency, plot_intensity_preference, plot_policy_evolution, plot_test_scores_vs_budget, plot_val_scores_vs_budget
from plotting.utils import add_master_augmentation_legend, add_master_strategy_legend, generate_dynamic_palette, load_and_flatten_agent_data, load_experiment_comparison_data, load_multibudget_agent_data, load_multibudget_agent_scores, load_multibudget_test_scores, load_multibudget_val_scores, load_single_budget_agent_scores, make_aug_int_label, prepare_augmentation_data, sort_key


def plot_agent_strategy_dashboard(config: ExperimentConfig, budget_dir: Path, save_fig: bool = True):
    save_path = budget_dir / "agent_strategy_dashboard.png"
    
    df = load_and_flatten_agent_data(budget_dir)
    df_scores = load_single_budget_agent_scores(budget_dir)
    
    if df.empty:
        print(f"No valid agent data found in {budget_dir}. Skipping plot.")
        return
        
    df['progress'] = df.groupby(['run'])['step'].transform(lambda x: x / x.max())
    total_epochs: int = config.training.active_epochs
    df['epoch'] = np.ceil(df['progress'] * total_epochs).astype(int).replace(0, 1) # type: ignore
    
    if not df_scores.empty:
        df_scores['progress'] = df_scores.groupby(['run'])['step'].transform(lambda x: x / x.max())
        df_scores['epoch'] = np.ceil(df_scores['progress'] * total_epochs).astype(int).replace(0, 1) # type: ignore

    df_active = df[df['augmentation'] != 'DoNothing'].copy()
    unique_augs = df_active['augmentation'].unique().tolist()
    global_order = ['DoNothing'] + sorted(unique_augs, reverse=True) if 'DoNothing' in df['augmentation'].values else sorted(unique_augs, reverse=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(18, 12)) 
    
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax_time = fig.add_subplot(gs[0, 0])
    ax_conf = fig.add_subplot(gs[1, 0])

    ax_int = fig.add_subplot(gs[0, 1])
    ax_freq = fig.add_subplot(gs[1, 1])

    plot_policy_evolution(ax_time, df_active, AUGMENTATION_COLORS, get_intensity_shade, time_col='epoch', group_cols=['run'])
    ax_time.set_xticks(range(1, total_epochs + 1, max(1, total_epochs // 10)))
    ax_time.set_xlabel("") 
    
    plot_agent_confidence(ax_conf, df_scores, time_col='epoch')
    ax_conf.set_xticks(range(1, total_epochs + 1, max(1, total_epochs // 10)))
    ax_conf.set_xlabel("Training Epoch")
    
    plot_intensity_preference(ax_int, df, global_order, AUGMENTATION_COLORS, get_intensity_shade)
    
    plot_global_active_frequency(ax_freq, df_active, AUGMENTATION_COLORS, global_order, group_cols=['run'], set_title=False)

    df_active['aug_with_intensity'] = df_active.apply(make_aug_int_label, axis=1)
    
    add_master_augmentation_legend(
        fig=fig, 
        df_active=df_active, 
        palette=AUGMENTATION_COLORS, 
        shade_func=get_intensity_shade, 
        sort_key=sort_key
    )
    
    strategy = config.augmentation_strategy.agent.value if config.augmentation_strategy.agent else "None"
    fig.suptitle(f"Active Learning Agent Dashboard ({strategy}) \n{config.model.name.value} ({config.model.weights_status.value} weights) on {config.dataset.name.value}", fontsize=20, fontweight='black', y=0.96)
        
    plt.subplots_adjust(top=0.88, bottom=0.15) 
    
    if save_fig:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Agent strategy dashboard saved successfully to {save_path}")


def generate_strategy_dashboard(config: ExperimentConfig, exp_root_dir: Path, save_fig: bool = True):
    save_path = exp_root_dir / "strategy_dashboard.png"
    
    df = load_multibudget_agent_data(exp_root_dir)
    df_val = load_multibudget_val_scores(exp_root_dir)
    df_test = load_multibudget_test_scores(exp_root_dir)
    df_scores = load_multibudget_agent_scores(exp_root_dir)
    
    strategy = config.augmentation_strategy.agent.value if config.augmentation_strategy.agent else "None"
    suptitle_text = f"Active Learning Strategy Dashboard ({strategy}) \n{config.model.name.value} ({config.model.weights_status.value} weights) on {config.dataset.name.value}"
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    if df.empty:
        fig = plt.figure(figsize=(20, 7))
        gs = gridspec.GridSpec(1, 2, wspace=0.20)
        
        ax_val = fig.add_subplot(gs[0, 0])
        ax_test = fig.add_subplot(gs[0, 1])
        
        plot_val_scores_vs_budget(ax_val, df_val)
        plot_test_scores_vs_budget(ax_test, df_test)
        
        sns.despine(fig=fig)
        fig.suptitle(suptitle_text, fontsize=20, fontweight='black', y=1.05)
        plt.subplots_adjust(top=0.85)
        
        if save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close(fig)
            print(f"Partial Strategy Dashboard saved to {save_path}")
        return

    df['progress'] = df.groupby(['budget', 'run'])['step'].transform(lambda x: x / x.max())
    total_epochs = config.training.active_epochs
    df['normalized_epoch'] = np.ceil(df['progress'] * total_epochs).astype(int).replace(0, 1)  # type: ignore
    
    if not df_scores.empty:
        df_scores['progress'] = df_scores.groupby(['budget', 'run'])['step'].transform(lambda x: x / x.max())
        df_scores['normalized_epoch'] = np.ceil(df_scores['progress'] * total_epochs).astype(int).replace(0, 1) # type: ignore

    df_active = df[df['augmentation'] != 'DoNothing'].copy()
    unique_augs = df_active['augmentation'].unique().tolist()
    global_order = ['DoNothing'] + sorted(unique_augs, reverse=True) if 'DoNothing' in df['augmentation'].values else sorted(unique_augs, reverse=True)

    fig = plt.figure(figsize=(20, 12)) 
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.20)

    ax_val = fig.add_subplot(gs[0, 0])
    ax_test = fig.add_subplot(gs[0, 1])
    
    gs_bottom_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[4, 2], hspace=0.35)
    ax_evol = fig.add_subplot(gs_bottom_left[0])
    ax_conf = fig.add_subplot(gs_bottom_left[1])

    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[4, 2], hspace=0.35)
    ax_int = fig.add_subplot(gs_bottom_right[0])
    ax_freq = fig.add_subplot(gs_bottom_right[1])

    plot_val_scores_vs_budget(ax_val, df_val)
    plot_test_scores_vs_budget(ax_test, df_test)
    
    plot_policy_evolution(ax_evol, df_active, AUGMENTATION_COLORS, get_intensity_shade, time_col='normalized_epoch', group_cols=['budget', 'run'])
    ax_evol.set_xticks(range(1, total_epochs + 1, max(1, total_epochs // 10)))
    # ax_evol.set_xlabel("Normalized Training Epoch")
    
    plot_agent_confidence(ax_conf, df_scores, time_col='normalized_epoch')
    ax_conf.set_xticks(range(1, total_epochs + 1, max(1, total_epochs // 10)))
    
    plot_global_active_frequency(ax_freq, df_active, AUGMENTATION_COLORS, global_order, group_cols=['budget', 'run'], set_title=False)
    
    plot_intensity_preference(ax_int, df, global_order, AUGMENTATION_COLORS, get_intensity_shade)

    df_active['aug_with_intensity'] = df_active.apply(make_aug_int_label, axis=1)
    add_master_augmentation_legend(
        fig=fig, 
        df_active=df_active, 
        palette=AUGMENTATION_COLORS, 
        shade_func=get_intensity_shade, 
        sort_key=sort_key
    )
    
    sns.despine(fig=fig)
    fig.suptitle(suptitle_text, fontsize=20, fontweight='black', y=0.96)
    plt.subplots_adjust(top=0.91, bottom=0.18) 
    
    if save_fig:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)
        print(f"Strategy Dashboard saved to {save_path}")

def generate_experiment_comparison(root_dir: Path, save_fig: bool = True, show_fig: bool = False):
    save_path = root_dir / "experiment_comparison_dashboard.png"
    baselines = ['none', 'random', 'static', 'trivialaugment', 'randaugment', 'autoaugment']
    
    df_metrics, df_test, _, df_select, _, df_scores, config_file = load_experiment_comparison_data(root_dir)
    
    if df_metrics.empty and df_test.empty:
        print("Missing required data (val/test metrics) for comparison dashboard.")
        return

    for df in [df_metrics, df_test, df_select, df_scores]:
        if not df.empty:
            df['strategy_display'] = df.apply(
                lambda x: f"{x['strategy']} (adv)" if x['trainer_type'] == 'adversarial' else x['strategy'], 
                axis=1
            )

    strats = set()
    for df in [df_metrics, df_test, df_select, df_scores]:
        if not df.empty and 'strategy_display' in df.columns:
            strats.update(df['strategy_display'].unique())
    strategy_order = sorted(list(strats))
    
    palette = generate_dynamic_palette(strategy_order)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    has_val = not df_metrics.empty
    has_test = not df_test.empty
    has_heatmaps = not df_select.empty
    
    if not has_val and not has_test and not has_heatmaps:
        print("No valid data found for comparison dashboard.")
        return
    
    if has_heatmaps:
        fig = plt.figure(figsize=(20, 12)) 
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
        
        ax_val = fig.add_subplot(gs[0, 0])
        ax_test_overall = fig.add_subplot(gs[0, 1]) 
        
        gs_bottom_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[4, 1.5], hspace=0.15)
        ax_heat = fig.add_subplot(gs_bottom_left[0]) 
        ax_conf_heat = fig.add_subplot(gs_bottom_left[1]) 

        ax_test_clean = fig.add_subplot(gs[1, 1]) 
            
        if has_val: plot_comparison_val_metrics(ax_val, df_metrics, strategy_order, palette)
        else: ax_val.set_axis_off()
            
        if has_test: plot_comparison_test_metrics(ax_test_overall, ax_test_clean, df_test, strategy_order, palette, baselines)
        else:
            ax_test_overall.set_axis_off()
            ax_test_clean.set_axis_off()
            
        plot_comparison_heatmap(ax_heat, df_select, baselines, set_xticks=False)    
        plot_comparison_confidence_heatmap(ax_conf_heat, df_scores, df_select, baselines)

        legend_ax = ax_test_overall if has_test else ax_val
    else:
        num_cols = (1 if has_val else 0) + (2 if has_test else 0)
        fig = plt.figure(figsize=(8 * num_cols, 6))
        gs = fig.add_gridspec(1, num_cols, wspace=0.25)
        
        col_idx = 0
        if has_val:
            ax_val = fig.add_subplot(gs[0, col_idx])
            plot_comparison_val_metrics(ax_val, df_metrics, strategy_order, palette)
            legend_ax = ax_val
            col_idx += 1
            
        if has_test:
            ax_test_overall = fig.add_subplot(gs[0, col_idx])
            ax_test_clean = fig.add_subplot(gs[0, col_idx + 1])
            plot_comparison_test_metrics(ax_test_overall, ax_test_clean, df_test, strategy_order, palette, baselines)
            legend_ax = ax_test_overall

    add_master_strategy_legend(fig, legend_ax, title="Active Learning Strategy") # type: ignore
    
    if config_file and config_file.exists():
        config = load_config(str(config_file))
        fig.suptitle(f"Strategy Comparison\n{config.model.name.value} ({config.model.weights_status.value} weights) on {config.dataset.name.value}", fontsize=20, fontweight='black', y=0.96 if has_heatmaps else 1.05)
    else:
        fig.suptitle("Strategy Comparison", fontsize=20, fontweight='black', y=0.96 if has_heatmaps else 1.05)
    
    plt.subplots_adjust(top=0.88 if has_heatmaps else 0.80, bottom=0.2 if has_heatmaps else 0.25) 
    

    if save_fig:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        print(f"Comparison dashboard saved to {save_path}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)



def plot_augmentation_facet_grid(root_dir: Path, save_fig: bool = True, show_fig: bool = False):
    """
    Plots a dynamic grid of line charts with paired styling: 
    Triangles/Dashes for Adversarial, Circles/Solid for Standard.
    """
    save_path = root_dir / "augmentation_robustness_grid.png"
    baselines = ['none', 'random', 'static', 'trivialaugment', 'randaugment', 'autoaugment']

    df_metrics, _, df_test_augs, _, _, _, config_file = load_experiment_comparison_data(root_dir)
    
    def parse_strat(name):
        name_str = str(name)
        if name_str.endswith('_adv'):
            return name_str.replace('_adv', ''), 'Adversarial'
        return name_str, 'Standard'

    df_agg = prepare_augmentation_data(df_test_augs)
    df_agg = df_agg[df_agg['base_augmentation'] != 'DoNothing']
    
    parsed_info = df_agg['strategy'].apply(parse_strat)
    df_agg['base_strategy'] = [x[0] for x in parsed_info]
    df_agg['training_type'] = [x[1] for x in parsed_info]

    unique_base_strats = sorted(df_agg['base_strategy'].unique())
    palette = generate_dynamic_palette(unique_base_strats)
    
    df_agg['is_baseline'] = df_agg['base_strategy'].apply(lambda x: str(x).lower() in baselines)
    df_plot = df_agg.sort_values(by=['is_baseline', 'base_strategy', 'budget'], ascending=[False, True, True])
    
    marker_map = {'Adversarial': 'v', 'Standard': 'o'}
    dash_map = {'Adversarial': (3, 2), 'Standard': ""} # (3, 2) creates a dashed effect

    g = sns.relplot(
        data=df_plot,
        x='budget', 
        y='test_f1_mean', 
        hue='base_strategy',      # Color determined by base name
        style='training_type',    # Shape/Dash determined by training type
        markers=marker_map,
        dashes=dash_map,
        col='base_augmentation', 
        col_wrap=3,
        kind='line', 
        palette=palette,
        height=3.5, 
        aspect=1.2,
        facet_kws={'sharey': True},
        linewidth=2.5,
        markersize=8,             # Triangles need to be slightly larger to be visible
        markeredgecolor='white',
        markeredgewidth=1
    )
    
    g.set_axis_labels("Training Set Budget", "Test F1 Score")
    g.set_titles("{col_name}", fontweight='bold', fontsize=12)
    
    if g.legend:
        g.legend.set_title("Strategy & Type")

    if config_file and config_file.exists():
        config = load_config(str(config_file))
        g.figure.suptitle(
            f"Robustness Comparison: {config.model.name.value} on {config.dataset.name.value}", 
            fontsize=20, fontweight='black', y=0.98
        )
    else:
        g.figure.suptitle("Strategy Comparison", fontsize=20, fontweight='black', y=0.98)

    g.figure.subplots_adjust(top=0.85) 

    if save_fig:
        g.figure.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Augmentation_robustness_grid saved to {save_path}")

    if show_fig:
        plt.show()
        
    plt.close(g.figure)

# def plot_augmentation_facet_grid(root_dir: Path, save_fig: bool = True, show_fig: bool = False):
#     """
#     Plots a dynamic grid of line charts, one for each base augmentation type.
#     """
#     save_path = root_dir / "augmentation_robustness_grid.png"
#     baselines = ['none', 'random', 'static', 'trivialaugment', 'randaugment', 'autoaugment']

#     df_metrics, _, df_test_augs, _, _, _, config_file = load_experiment_comparison_data(root_dir)
    
#     strategy_order = sorted(df_metrics['strategy'].unique())
#     df_agg = prepare_augmentation_data(df_test_augs)
#     df_agg = df_agg[df_agg['base_augmentation'] != 'DoNothing']
#     palette = generate_dynamic_palette(strategy_order)
    
#     df_agg['is_baseline'] = df_agg['strategy'].apply(lambda x: str(x).lower() in baselines)
#     df_plot = df_agg.sort_values(by=['is_baseline', 'strategy', 'budget'], ascending=[False, True, True])
    
#     g = sns.relplot(
#         data=df_plot,
#         x='budget', 
#         y='test_f1_mean', 
#         hue='strategy',
#         col='base_augmentation', 
#         col_wrap=3,
#         kind='line', 
#         palette=palette,
#         height=3.5, 
#         aspect=1.2,
#         facet_kws={'sharey': True},
#         linewidth=2.5,
#         marker='o',
#         markersize=6,
#         markeredgecolor='white',
#         markeredgewidth=1
#     )
    
#     g.set_axis_labels("Training Set Budget", "Test F1 Score")
#     g.set_titles("{col_name}", fontweight='bold', fontsize=12)
    
    
#     if config_file and config_file.exists():
#         config = load_config(str(config_file))
#         g.figure.suptitle(f"Strategy Comparison\n{config.model.name.value} ({config.model.weights_status.value} weights) on {config.dataset.name.value}", fontsize=20, fontweight='black', y=0.96)
#     else:
#         g.figure.suptitle("Strategy Comparison", fontsize=20, fontweight='black', y=0.96)

#     g.figure.subplots_adjust(top=0.8) 

#     if save_fig:
#         g.figure.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
#         print(f"Augmentation_robustness_grid saved to {save_path}")

#     if show_fig:
#         plt.show()
        
#     plt.close(g.figure)