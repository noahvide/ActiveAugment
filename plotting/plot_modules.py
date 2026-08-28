import pandas as pd
import numpy as np
import seaborn as sns

from plotting.utils import make_aug_int_label, parse_strategy, sort_key


def plot_global_active_frequency(ax, df_active, palette, order, group_cols=['run'], set_title: bool = True):
    run_counts = df_active.groupby(group_cols + ['augmentation']).size().reset_index(name='count')
    run_counts['frequency_pct'] = run_counts.groupby(group_cols)['count'].transform(lambda x: (x / x.sum()) * 100)
    
    valid_order = [o for o in order if o in df_active['augmentation'].unique()]
    
    sns.barplot(
        data=run_counts, x='frequency_pct', y='augmentation', hue='augmentation',
        legend=False, ax=ax, order=valid_order, palette=palette,
        capsize=0.1, errorbar=("ci", 90)
    )
    
    if set_title:
        ax.set_title("Global Active Augmentation Frequency", fontweight='bold')
    ax.set_xlabel("Mean Selection Frequency (%)") 
    ax.set_ylabel("")

def plot_length_frequency(ax, df_lengths, palette, display_order, group_cols=['run']):
    len_counts = df_lengths.groupby(group_cols + ['length']).size().reset_index(name='count')
    len_counts['frequency_pct'] = len_counts.groupby(group_cols)['count'].transform(lambda x: (x / x.sum()) * 100)

    sns.barplot(
        data=len_counts, x='frequency_pct', y='length', hue='length', orient='h',
        legend=False, ax=ax, palette=palette, capsize=0.1, errorbar=("ci", 90), order=display_order
    )
    ax.set_xlabel("Mean Selection Frequency (%)")
    ax.set_ylabel("Pipeline\nLength", rotation=0, labelpad=25, va='center')

def plot_intensity_preference(ax, df, order, palette, shade_func):
    df_ints = df[~df['augmentation'].isin(['N/A', 'DoNothing', 'Flip'])].copy()
    
    if df_ints.empty:
        ax.text(0.5, 0.5, "No intensity data available", ha='center', va='center')
        ax.set_axis_off()
        return

    int_crosstab = pd.crosstab(df_ints['augmentation'], df_ints['intensity'], normalize='index') * 100
    valid_int_order = [o for o in order if o in int_crosstab.index]
    int_crosstab = int_crosstab.loc[valid_int_order]
    
    augmentations = int_crosstab.index
    present_intensities = [i for i in ['LOW', 'MEDIUM', 'HIGH', 'ALL'] if i in int_crosstab.columns]
    bottoms = np.zeros(len(augmentations))
    
    for intensity in present_intensities:
        bar_colors = [shade_func(palette.get(aug, '#808080'), intensity) for aug in augmentations]
        values = int_crosstab[intensity].values
        ax.bar(augmentations, values, bottom=bottoms, color=bar_colors, edgecolor='white', linewidth=1) 
        bottoms += values # type: ignore
        
    ax.set_title("Intensity Preference Proportion (%)", fontweight='bold')
    ax.set_ylabel("Proportion (%)")
    ax.tick_params(axis='x')

def plot_policy_evolution(ax, df_active, palette, shade_func, time_col='epoch', group_cols=None):
    df_time = df_active.copy()
    df_time['aug_with_intensity'] = df_time.apply(make_aug_int_label, axis=1)
    
    if group_cols:
        counts = df_time.groupby(group_cols + [time_col, 'aug_with_intensity']).size().unstack(fill_value=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        policy_pct = pct.groupby(time_col).mean()
    else:
        counts = df_time.groupby([time_col, 'aug_with_intensity']).size().unstack(fill_value=0)
        policy_pct = counts.div(counts.sum(axis=1), axis=0) * 100
        
    ordered_cols = sorted(policy_pct.columns, key=sort_key)

    policy_pct = policy_pct[ordered_cols]
    
    final_colors = []
    intensity_map = {'(H)': 'HIGH', '(M)': 'MEDIUM', '(L)': 'LOW'}
    for col in ordered_cols:
        aug, int_str = col.split(' (')[0], intensity_map.get(col[-3:], 'ALL')
        final_colors.append(shade_func(palette.get(aug, '#000000'), int_str))
        
    ax.stackplot(policy_pct.index, policy_pct.T, labels=policy_pct.columns, colors=final_colors)
    ax.set_title("Active Policy Evolution", fontweight='bold', fontsize=14)
    ax.set_ylabel("Selection Freq (%)")
    ax.set_ylim(0, 100)
    if policy_pct.index.min() != policy_pct.index.max(): ax.set_xlim(policy_pct.index.min(), policy_pct.index.max())

def plot_length_evolution(ax, df_lengths, palette, time_col='epoch', group_cols=None):
    if group_cols:
        len_evol = df_lengths.groupby(group_cols + [time_col, 'length']).size().unstack(fill_value=0)
        pct = len_evol.div(len_evol.sum(axis=1), axis=0) * 100
        len_pct = pct.groupby(time_col).mean()
    else:
        len_evol = df_lengths.groupby([time_col, 'length']).size().unstack(fill_value=0)
        len_pct = len_evol.div(len_evol.sum(axis=1), axis=0) * 100
        
    stack_colors = [palette[col] for col in len_pct.columns]
    ax.stackplot(len_pct.index, len_pct.T, labels=len_pct.columns, colors=stack_colors)
    ax.set_ylabel("Length %", rotation=0, labelpad=25, va='center')
    ax.set_ylim(0, 100)
    if len_pct.index.min() != len_pct.index.max(): ax.set_xlim(len_pct.index.min(), len_pct.index.max())

def plot_syntax_heatmap(ax, df, order):
    heat_data = pd.crosstab(df['augmentation'], df['stage'])
    valid_order = [o for o in order if o in heat_data.index]
    heat_data = heat_data.loc[valid_order]
    heat_pct = heat_data.div(heat_data.sum(axis=0), axis=1) * 100
    
    if 'DoNothing' in heat_pct.index:
        heat_pct = heat_pct.drop(index='DoNothing')
    
    sns.heatmap(heat_pct, ax=ax, cmap="Blues", annot=True, fmt=".1f", linewidths=.5, cbar_kws={'label': '% Frequency'})
    ax.set_title("Pipeline Syntax (Position Frequency)", fontweight='bold')
    ax.set_xlabel("Pipeline Stage")
    ax.set_ylabel("")
    ax.tick_params(axis='y', rotation=0)
    
    

def plot_val_scores_vs_budget(ax, df_val):
    if df_val.empty: return
    b_agg = df_val.groupby('budget')['score'].agg(['mean', 'std']).reset_index()
    
    ax.plot(b_agg['budget'], b_agg['mean'], marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1, linewidth=2, color='#2c3e50')
    ax.fill_between(b_agg['budget'], b_agg['mean'] - b_agg['std'], b_agg['mean'] + b_agg['std'], alpha=0.2, color='#2c3e50', linewidth=0)
    ax.set_title("Validation Score vs. Budget", fontweight='bold', fontsize=14)
    ax.set_xlabel("Dataset Budget")
    ax.set_ylabel("Final Score (Mean ± Std)")
    # ax.set_xticks(b_agg['budget'])

def plot_test_scores_vs_budget(ax, df_test):
    if df_test.empty: return
    metric_name = df_test['metric_name'].iloc[0]
    
    ax.plot(df_test['budget'], df_test['overall_mean'], marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1, linewidth=2, color='#2c3e50', label='Overall Test')
    ax.fill_between(df_test['budget'], df_test['overall_mean'] - df_test['overall_std'], df_test['overall_mean'] + df_test['overall_std'], alpha=0.2, color='#2c3e50', linewidth=0)
    
    ax.plot(df_test['budget'], df_test['clean_mean'], marker='s', markersize=6, markeredgecolor='white', markeredgewidth=1, linewidth=2, color='#27ae60', label='Clean/Original Test')
    ax.fill_between(df_test['budget'], df_test['clean_mean'] - df_test['clean_std'], df_test['clean_mean'] + df_test['clean_std'], alpha=0.2, color='#27ae60', linewidth=0)

    ax.set_title(f"Test {metric_name} vs. Budget", fontweight='bold', fontsize=14)
    ax.set_xlabel("Dataset Budget")
    ax.set_ylabel(f"Test {metric_name} (Mean ± Std)")
    # ax.set_xticks(df_test['budget'])
    ax.legend(loc='lower right', frameon=True)

def plot_preference_rank(ax, df_active, palette, shade_func):
    df_active['aug_with_intensity'] = df_active.apply(make_aug_int_label, axis=1)
    budget_counts = df_active.groupby(['budget', 'aug_with_intensity']).size().unstack(fill_value=0)
    budget_ranks = budget_counts.rank(axis=1, ascending=False, method='min')
    
    intensity_map = {'H': 'HIGH', 'M': 'MEDIUM', 'L': 'LOW', 'A': 'ALL'}
    for col in budget_ranks.columns:
        parts = col.split(' (')
        aug_name, int_short = parts[0], parts[1][0] if len(parts) > 1 else 'A'
        base_color = palette.get(aug_name, '#000000')
        shaded_color = shade_func(base_color, intensity_map.get(int_short, 'ALL'))
        ax.plot(budget_ranks.index, budget_ranks[col], marker='o', color=shaded_color, linewidth=2.5, markersize=8)
        
    ax.set_ylabel("Preference Rank", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    num_options = len(budget_ranks.columns)
    ax.set_yticks(range(1, num_options + 1))
    ax.set_ylim(0.5, num_options + 0.5)
    ax.invert_yaxis() 
    ax.set_xticks(budget_ranks.index)
    ax.set_title("Active Agent Preference Rank vs. Dataset Budget", fontweight='bold', fontsize=14)
    ax.set_xlabel("Dataset Budget", fontweight='bold')


# def plot_comparison_val_metrics(ax, df_metrics, strategy_order, palette):
#     df_metrics['base_strategy'] = df_metrics['strategy'].apply(lambda x: parse_strategy(x)[0])
#     df_metrics['training_type'] = df_metrics['strategy'].apply(lambda x: 'Adversarial' if parse_strategy(x)[1] else 'Standard')
    
#     marker_map = {'Adversarial': "v", 'Standard': "o"}
    
#     base_order = [parse_strategy(s)[0] for s in strategy_order]
    
#     sns.lineplot(
#         data=df_metrics, x='budget', y='val_f1', 
#         hue='base_strategy', hue_order=base_order, palette=palette, 
#         style='training_type', markers=marker_map,
#         markersize=8, markeredgecolor='white', markeredgewidth=0.5,
#         linewidth=2, ax=ax, errorbar=("ci", 90)
#     )
    
#     ax.set_title("Training: Val F1 vs. Budget", fontweight='bold', fontsize=14)
#     if ax.get_legend(): ax.get_legend().remove()
    
    # Clean up legend to show both Strategy (Colors) and Type (Markers)
    # ax.legend(title="Strategy & Type", bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_comparison_val_metrics(ax, df_metrics, strategy_order, palette):
    marker_map = {"standard": "o", "adversarial": "^", "contrastive": ">", "maxup": "s"}
    
    sns.lineplot(
        data=df_metrics, 
        x='budget', 
        y='val_f1', 
        hue='strategy_display',
        style='trainer_type',
        hue_order=strategy_order, 
        palette=palette, 
        markers=marker_map,
        dashes=False,
        markersize=8, 
        markeredgecolor='white', 
        markeredgewidth=0.5,
        linewidth=2, 
        ax=ax, 
        errorbar=("ci", 90)
    )
    
    ax.set_title("Training: Val F1 vs. Budget", fontweight='bold', fontsize=14)
    ax.set_xlabel("Training Set Budget")
    ax.set_ylabel("Validation F1 Score (Mean ± Std)")
    
    if ax.get_legend():
        ax.get_legend().remove()


def plot_comparison_test_metrics(ax_overall, ax_clean, df_test, strategy_order, palette, baselines):
    metric_name = "F1"
    marker_map = {"standard": "o", "adversarial": "^", "contrastive": ">", "maxup": "s"}
    
    for strat_display in strategy_order:
        strat_data = df_test[df_test['strategy_display'] == strat_display].sort_values('budget')
        
        if strat_data.empty:
            continue
        
        trainer_type = strat_data['trainer_type'].iloc[0]
        marker = marker_map.get(trainer_type, 's') # Default to square if type is unknown
        linestyle = '-'
        
        color = palette.get(strat_display, '#333333')
        
        base_name = strat_display.replace(" (adv)", "").lower()
        z_order = 1 if base_name in baselines else 2
        
        ax_overall.plot(
            strat_data['budget'], strat_data['test_f1_overall_mean'], 
            marker=marker, markersize=8, markeredgecolor='white', markeredgewidth=1, 
            color=color, linestyle=linestyle, linewidth=2, label=strat_display, zorder=z_order
        )
        ax_overall.fill_between(
            strat_data['budget'], 
            strat_data['test_f1_overall_mean'] - strat_data['test_f1_overall_std'], 
            strat_data['test_f1_overall_mean'] + strat_data['test_f1_overall_std'], 
            color=color, alpha=0.15, linewidth=0, zorder=z_order
        ) 
        
        ax_clean.plot(
            strat_data['budget'], strat_data['test_f1_clean_mean'], 
            marker=marker, markersize=8, markeredgecolor='white', markeredgewidth=1, 
            color=color, linestyle=linestyle, linewidth=2, label=strat_display, zorder=z_order
        )
        ax_clean.fill_between(
            strat_data['budget'], 
            strat_data['test_f1_clean_mean'] - strat_data['test_f1_clean_std'], 
            strat_data['test_f1_clean_mean'] + strat_data['test_f1_clean_std'], 
            color=color, alpha=0.15, linewidth=0, zorder=z_order
        ) 

    ax_overall.set_title(f"Overall Test {metric_name} vs. Budget", fontweight='bold', fontsize=14)
    ax_overall.set_xlabel("Training Set Budget")
    ax_overall.set_ylabel(f"Overall Test {metric_name} (Mean ± Std)")
    
    ax_clean.set_title(f"Clean/Original Test {metric_name} vs. Budget", fontweight='bold', fontsize=14)
    ax_clean.set_xlabel("Training Set Budget")
    ax_clean.set_ylabel(f"Clean Test {metric_name} (Mean ± Std)")


# def plot_comparison_test_metrics(ax_overall, ax_clean, df_test, strategy_order, palette, baselines):
#     metric_name = "F1"
    
#     for strat in strategy_order:
#         strat_data = df_test[df_test['strategy'] == strat].sort_values('budget')
#         color = palette.get(strat, '#333333')
#         z_order = 1 if strat.lower() in baselines else 2
        
#         ax_overall.plot(strat_data['budget'], strat_data['test_f1_overall_mean'], marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1, color=color, linestyle='-', linewidth=2, label=strat, zorder=z_order)
#         ax_overall.fill_between(strat_data['budget'], strat_data['test_f1_overall_mean'] - strat_data['test_f1_overall_std'], strat_data['test_f1_overall_mean'] + strat_data['test_f1_overall_std'], color=color, alpha=0.2, linewidth=0, zorder=z_order) 
        
#         ax_clean.plot(strat_data['budget'], strat_data['test_f1_clean_mean'], marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1, color=color, linestyle='-', linewidth=2, label=strat, zorder=z_order)
#         ax_clean.fill_between(strat_data['budget'], strat_data['test_f1_clean_mean'] - strat_data['test_f1_clean_std'], strat_data['test_f1_clean_mean'] + strat_data['test_f1_clean_std'], color=color, alpha=0.2, linewidth=0, zorder=z_order) 

#     ax_overall.set_title(f"Overall Test {metric_name} vs. Budget", fontweight='bold', fontsize=14)
#     ax_overall.set_xlabel("Training Set Budget")
#     ax_overall.set_ylabel(f"Overall Test {metric_name} (Mean ± Std)")
    
#     ax_clean.set_title(f"Clean/Original Test {metric_name} vs. Budget", fontweight='bold', fontsize=14)
#     ax_clean.set_xlabel("Training Set Budget")
#     ax_clean.set_ylabel(f"Clean Test {metric_name} (Mean ± Std)")

def plot_comparison_lengths(ax, df_len, strategy_order, palette):
    sns.lineplot(data=df_len, x='budget', y='length', hue='strategy', 
                 hue_order=strategy_order, palette=palette, 
                 marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1,
                 linewidth=2, ax=ax, errorbar=("ci", 90))
    ax.set_title("Strategy: Average Pipeline Length vs. Budget", fontweight='bold', fontsize=14)
    ax.set_xlabel("Training Set Budget")
    ax.set_ylabel("Active Pipeline Length (Mean ± Std)")
    if ax.get_legend(): ax.get_legend().remove()


def plot_comparison_heatmap(ax, df_select, baselines, set_xticks: bool = True):
    exclude_base = ['RandAugment', 'AutoAugment', 'TrivialAugment']

    df_filtered = df_select[~df_select['strategy'].isin(exclude_base)]
    
    if df_filtered.empty:
        ax.set_axis_off()
        return
    
    dna_data = df_filtered.groupby(['strategy_display', 'augmentation']).size().unstack(fill_value=0)
    dna_pct = dna_data.div(dna_data.sum(axis=1), axis=0) * 100
    
    cols = sorted([col for col in dna_pct.columns if col not in exclude_base])
    dna_pct = dna_pct[cols]
    
    sorted_index = sorted(
        dna_pct.index, 
        key=lambda x: (x.replace(" (adv)", "").lower() in baselines, x)
    )
    
    dna_pct_transposed = dna_pct.loc[sorted_index].T
    
    if not dna_pct_transposed.empty:
        sns.heatmap(
            dna_pct_transposed, 
            annot=True, 
            fmt=".1f", 
            cmap="Blues", 
            ax=ax, 
            cbar_kws={'label': '% of Active Selections'}
        )
        
        ax.set_title("Augmentation Selection Frequency", fontweight='bold', fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        if not set_xticks:
            ax.set_xticks([])
        else:
            ax.tick_params(axis='x', rotation=45)
    else:
        # Optional: Handle the empty state (e.g., clear the axis or add a placeholder text)
        ax.set_title("No Data Available for Augmentation", fontsize=10, fontstyle='italic')
        ax.set_xticks([])
        ax.set_yticks([])

    
def plot_comparison_length_heatmap(ax, df_len, baselines):
    if df_len.empty: return

    mean_len = df_len.groupby('strategy')['length'].mean().to_frame(name='Mean Length')

    sorted_index = sorted(mean_len.index, key=lambda x: (x.lower() in baselines, x))
    mean_len_transposed = mean_len.loc[sorted_index].T

    sns.heatmap(
        mean_len_transposed, 
        annot=True, 
        fmt=".2f", 
        cmap="crest",
        ax=ax, 
        cbar_kws={'label': 'Length'},
        linewidths=0.5
    )
    
    ax.set_xlabel("Strategy", fontweight='bold')
    ax.set_ylabel("")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')


def plot_agent_confidence(ax, df_scores, time_col='normalized_epoch'):
    """Plots the agent's internal confidence metrics over time as a Coefficient of Variation."""
    if df_scores.empty:
        ax.set_axis_off()
        return

    df_plot = df_scores.copy()
    epsilon = 1e-8
    
    df_plot['cv_std_dev'] = df_plot['mean_std_dev'] / (df_plot['mean_batch_score'].abs() + epsilon)
    df_plot['cv_margin'] = df_plot['mean_margin'] / (df_plot['mean_batch_score'].abs() + epsilon)

    id_vars = [time_col]
    if 'budget' in df_plot.columns:
        id_vars.append('budget')
    if 'run' in df_plot.columns:
        id_vars.append('run')

    df_melted = df_plot.melt(
        id_vars=id_vars,
        value_vars=['cv_std_dev', 'cv_margin'],
        var_name='Metric', 
        value_name='Value'
    )

    df_melted['Metric'] = df_melted['Metric'].map({
        'cv_std_dev': 'Spread CV (Std/Mean)',
        'cv_margin': 'Margin CV (Margin/Mean)'
    })

    sns.lineplot(
        data=df_melted, 
        x=time_col, 
        y='Value', 
        hue='Metric',
        errorbar=('ci', 90), 
        ax=ax, 
        palette=['#FF6B6B', '#4ECDC4'],
        linewidth=2
    )

    ax.set_xlim(df_melted[time_col].min(), df_melted[time_col].max())

    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.set_xlabel("Normalized Training Epoch")
    
    ax.legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, framealpha=0.0)


def plot_comparison_confidence_heatmap(ax, df_scores, df_select, baselines):
    """Plots a summary heatmap of agent confidence metrics (CV) per strategy."""
    if df_scores.empty or df_select.empty:
        ax.set_axis_off()
        return

    exclude_strats = ['RandAugment', 'TrivialAugment', 'AutoAugment']

    df_plot = df_scores[~df_scores['strategy'].isin(exclude_strats)].copy()
    
    if df_plot.empty:
        ax.set_axis_off()
        return
    
    epsilon = 1e-8
    df_plot['cv_std_dev'] = df_plot['mean_std_dev'] / (df_plot['mean_batch_score'].abs() + epsilon)
    df_plot['cv_margin'] = df_plot['mean_margin'] / (df_plot['mean_batch_score'].abs() + epsilon)

    agg_df = df_plot.groupby('strategy_display')[['cv_std_dev', 'cv_margin']].mean()
    
    sorted_active_strategies = sorted(
        agg_df.index, 
        key=lambda x: (x.replace(" (adv)", "").lower() in baselines, x)
    )
    
    agg_df = agg_df.reindex(sorted_active_strategies)
    
    agg_df.columns = ['Spread CV\n(Std / Mean)', 'Margin CV\n(Margin / Mean)']

    sns.heatmap(
        agg_df.T, 
        annot=True, 
        cmap="mako_r",
        fmt=".2f",
        cbar=True, 
        ax=ax,
        linewidths=.5
    )
    
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Aggregate Agent Confidence (CV)", fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
