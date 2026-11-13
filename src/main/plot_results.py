"""Plot saved report results for single-map or multimap experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from matplotlib.ticker import FuncFormatter

def final_load_results(directory, checkpoint='best'):
    rows = pd.read_csv(Path(directory) / 'final_test_episodes.csv')
    rows = rows.loc[rows['checkpoint'] == checkpoint].copy().sort_values('seed')
    if rows.empty or rows['seed'].duplicated().any():
        raise ValueError(f'Missing or duplicate best-checkpoint test seeds: {directory}')
    for key in ('terminated', 'truncated'):
        rows[key] = rows[key].astype(str).str.lower().eq('true')
    return rows.set_index('seed')

def final_plot_comparison(dagger, rlpd, output_dir, dagger_label='DAgger (loaded offline)', rlpd_label='RLPD', expert=None):
    data = {dagger_label: final_load_results(dagger), rlpd_label: final_load_results(rlpd)}
    left, right = data.values()
    if not left.index.equals(right.index):
        raise ValueError('Both actors must have exactly the same final-test seeds')
    expert_mean = None
    if expert is not None:
        baseline = final_load_results(expert, checkpoint='team4')
        if not baseline.index.equals(left.index):
            raise ValueError('Expert and actors must have exactly the same final-test seeds')
        if not baseline['terminated'].all():
            raise ValueError('Expert baseline requires completed laps on all test seeds')
        expert_mean = float(baseline['n_steps'].mean())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = left.index.to_numpy()
    x = np.arange(len(seeds))
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 15, 'axes.labelsize': 12, 'svg.fonttype': 'none'})
    colors = ['#4C72B0', '#DD8452']
    summaries, distributions = ([], [])
    fig, ax = plt.subplots(figsize=(15, 4.8))
    for (name, rows), color in zip(data.items(), colors):
        steps = rows['n_steps'].to_numpy(dtype=float)
        completed = rows['terminated'].to_numpy()
        mean, std = (steps.mean(), steps.std(ddof=1) if len(steps) > 1 else np.nan)
        ax.plot(x, steps, marker='o', markersize=5, color=color, linewidth=1.6, label=f"{('DAgger' if name == dagger_label else 'RLPD')}: {mean:.2f} ± {std:.2f} steps")
        ax.scatter(x[~completed], steps[~completed], color=color, marker='x', s=100)
        ax.axhline(mean, color=color, linestyle='--', linewidth=1.2)
        ax.axhspan(mean - std, mean + std, color=color, alpha=0.12)
        distributions.append(steps)
        finish = steps[completed]
        summaries.append(dict(method=name, episodes=len(steps), completed=int(completed.sum()), completion_rate=float(completed.mean()), steps_mean=mean, steps_std=std, finish_steps_mean=finish.mean() if len(finish) else np.nan, finish_steps_std=finish.std(ddof=1) if len(finish) > 1 else np.nan))
    ax.set(title=f'Steps to finish across {len(seeds)} test seeds - mean ± standard deviation', xlabel='Test seed', ylabel='Episode steps')
    ax.set_xticks(x, [str(seed) for seed in seeds], rotation=45, ha='right')
    all_steps = np.concatenate(distributions)
    ax.set_ylim(all_steps.min() - 10, all_steps.max() + 10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.045), ncol=2, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for suffix in ('svg',):
        path = output / f'dagger_rlpd_paired_episodes.{suffix}'
        fig.savefig(path, bbox_inches='tight')
        print(path)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    box = ax.boxplot(distributions, tick_labels=list(data), patch_artist=True, showfliers=False, widths=0.28, medianprops=dict(color='#222222', linewidth=2), whiskerprops=dict(color='#333333', linewidth=1.5), capprops=dict(color='#333333', linewidth=1.5))
    offsets = np.linspace(-0.1, 0.1, len(seeds))
    for index, (steps, color, patch) in enumerate(zip(distributions, colors, box['boxes']), start=1):
        patch.set(facecolor=color, edgecolor=color, alpha=0.6)
        ax.scatter(index + offsets, steps, color=color, edgecolor='white', linewidth=0.5, alpha=0.9, s=34, zorder=3)
    if expert_mean is not None:
        ax.axhline(expert_mean, color='#333333', linestyle='--', linewidth=1.4, label=f'Expert (team4) mean: {expert_mean:.2f} steps')
        ax.legend(frameon=False, fontsize=10)
    ax.set(title=f'Distribution of episode steps across {len(seeds)} test seeds', ylabel='Episode steps')
    fig.tight_layout()
    for suffix in ('svg',):
        path = output / f'dagger_rlpd_step_distribution.{suffix}'
        fig.savefig(path, bbox_inches='tight')
        print(path)
    plt.close(fig)
    both = left['terminated'] & right['terminated']
    values = (right['n_steps'] - left['n_steps']).where(both).to_numpy(dtype=float)
    pd.DataFrame(summaries).to_csv(output / 'dagger_rlpd_final_summary.csv', index=False)
    pd.DataFrame(dict(seed=seeds, dagger_steps=left['n_steps'].to_numpy(), rlpd_steps=right['n_steps'].to_numpy(), both_completed=both.to_numpy(), rlpd_minus_dagger_steps=values)).to_csv(output / 'dagger_rlpd_final_paired.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

def dagger_plot_results(experiment, output_dir=None):
    experiment = Path(experiment)
    output = Path(output_dir) if output_dir is not None else experiment / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, name):
        fig.tight_layout()
        path = output / name
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)
        print(path)
    loss_path = experiment / 'losses.csv'
    if loss_path.exists():
        losses = pd.read_csv(loss_path)
        if not losses.empty:
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
            for ax, column in zip(axes, ['mse', 'bce', 'loss']):
                ax.plot(np.arange(1, len(losses) + 1), losses[column], marker='.')
                for start in np.flatnonzero(losses['iteration'].diff().fillna(0).to_numpy() != 0):
                    ax.axvline(start + 0.5, color='grey', alpha=0.3, linewidth=1)
                ax.set(title=column.upper(), xlabel='Logged training epoch', ylabel='Loss')
                ax.grid(alpha=0.2)
            fig.suptitle('Training loss - vertical lines separate DAgger iterations')
            save(fig, 'losses.svg')
    evaluation = experiment / 'evaluation.csv'
    if not evaluation.exists():
        print('No evaluation.csv yet; evaluation plots skipped.')
        return saved
    data = pd.read_csv(evaluation)
    if data.empty:
        return saved
    if 'track' not in data:
        data['track'] = 'training_track'
    baseline_path = experiment / 'expert_benchmark' / 'summary.csv'
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    tracks = list(dict.fromkeys(data['track']))
    fig, axes = plt.subplots(3, len(tracks), figsize=(4.5 * len(tracks), 10), squeeze=False)
    for col, track in enumerate(tracks):
        rows = data[data['track'] == track].sort_values('iteration')
        expert = baseline[baseline['track'] == track] if not baseline.empty else baseline
        diagnostic = 'selection_track' in rows and (not rows['selection_track'].astype(str).str.lower().eq('true').any())
        axes[0, col].set_title(f'{track}' + (' (unseen)' if diagnostic else ''))
        for index, (metric, std, label) in enumerate([('completion_rate', None, 'Completion rate'), ('steps_mean', 'steps_std', 'Steps - all episodes'), ('finish_steps_mean', 'finish_steps_std', 'Steps - completed episodes')]):
            ax = axes[index, col]
            x = rows['iteration'].to_numpy()
            y = rows[metric].to_numpy(dtype=float)
            ax.plot(x, y, marker='o', label='Actor')
            if std in rows:
                spread = rows[std].to_numpy(dtype=float)
                ax.fill_between(x, y - spread, y + spread, alpha=0.2)
            if not expert.empty and pd.notna(expert.iloc[0][metric]):
                level = float(expert.iloc[0][metric])
                ax.axhline(level, color='tab:orange', linestyle='--', label='Expert')
                if std in expert and pd.notna(expert.iloc[0][std]):
                    spread = float(expert.iloc[0][std])
                    ax.axhspan(level - spread, level + spread, color='tab:orange', alpha=0.12)
            ax.set(xlabel='DAgger iteration', ylabel=label)
            if metric == 'completion_rate':
                ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle('Actor versus expert - bands show episode standard deviation, not confidence intervals')
    save(fig, 'evaluation_by_track.svg')
    if 'relative_finish_steps' in data:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for track in tracks:
            rows = data[data['track'] == track].sort_values('iteration')
            ax.plot(rows['iteration'], rows['relative_finish_steps'], marker='o', label=track)
        ax.axhline(1, linestyle='--', color='black', label='Expert reference')
        ax.set(xlabel='DAgger iteration', ylabel='Actor / expert mean finishing steps', title='Relative speed - completed episodes only; lower is faster')
        ax.grid(alpha=0.2)
        ax.legend()
        save(fig, 'relative_steps.svg')
    aggregate_path = experiment / 'evaluation_aggregate.csv'
    if aggregate_path.exists():
        rows = pd.read_csv(aggregate_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, label in zip(axes, ['completion_rate', 'relative_finish_steps'], ['Mean training-map completion rate', 'Mean training-map step ratio']):
            ax.plot(rows['training_steps'], rows[metric], marker='o')
            ax.set(xlabel='Cumulative collection steps', ylabel=label)
            ax.grid(alpha=0.2)
        save(fig, 'training_maps_aggregate.svg')
    return saved

def dagger_plot_relative_comparison(protocol, seeds, output_dir=None, relative=True):
    """Validation learning curves for each method's validation-selected best run.

    Bands use episode std / fixed expert mean, not training-seed variation.
    """
    protocol = Path(protocol)
    output = Path(output_dir) if output_dir is not None else protocol.parent / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    selected = {}
    for method in ('dagger_offline_gaussian', 'rlpd'):
        candidates = []
        for seed in seeds:
            folder = protocol / f'{method}_seed{seed}'
            best = json.loads((folder / 'best_evaluation.json').read_text())
            candidates.append(((-best['completion_rate'], best['finish_steps_mean'] if best['finish_steps_mean'] is not None else np.inf, seed), folder))
        selected[method] = min(candidates)[1]
    dagger = selected['dagger_offline_gaussian']
    config = json.loads((dagger / 'config.json').read_text())
    expert = pd.read_csv(dagger / 'expert_benchmark' / 'summary.csv')
    reference = float(expert.loc[expert['track'] == config['track'], 'finish_steps_mean'].iloc[0])
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(14, 3.8) if not relative else (11, 4.8))
    scale = reference if relative else 1.0
    stem = 'relative_steps_comparison' if relative else 'finishing_steps_comparison'
    title = 'Relative finishing steps' if relative else 'Finishing steps'
    ax.axhline(reference / scale, color='#333333', linestyle='--', linewidth=1.5, label='Expert (team4)')
    summaries = []
    for method, label, color in [('dagger_offline_gaussian', 'DAgger', '#4C72B0'), ('rlpd', 'RLPD', '#DD8452')]:
        folder = selected[method]
        settings = json.loads((folder / 'config.json').read_text())
        if any((settings[k] != config[k] for k in ('track', 'step_size', 'max_steps'))) or settings['evaluation']['seeds'] != config['evaluation']['seeds']:
            raise ValueError('Comparison requires matching task and evaluation seeds')
        rows = pd.read_csv(folder / 'evaluation.csv')
        column = 'online_steps' if 'online_steps' in rows else 'env_steps'
        rows = rows.sort_values(column)
        x = rows[column].to_numpy(dtype=float)
        y = rows['finish_steps_mean'].to_numpy(dtype=float) / scale
        std = rows['finish_steps_std'].to_numpy(dtype=float) / scale
        ax.plot(x, y, marker='o', markersize=4, color=color, label=label)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.15)
        for i, (_, row) in enumerate(rows.iterrows()):
            summaries.append(dict(method=label, source=str(folder), online_steps=x[i], mean_steps=row['finish_steps_mean'], std_steps=row['finish_steps_std'], relative_steps=row['finish_steps_mean'] / reference, relative_std=row['finish_steps_std'] / reference, expert_mean_steps=reference, completion_rate=row['completion_rate']))
    ax.set(xlabel='Online collection steps', ylabel=title, title=title)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ('svg',):
        path = output / f'{stem}.{suffix}'
        fig.savefig(path, bbox_inches='tight')
        print(path)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(output / f'{stem}.csv', index=False)
    return output / f'{stem}.svg'

def dagger_plot_initialization_details(protocol, seeds, output_dir=None):
    """All training-seed histories, then paired evaluation of selected best actors."""
    protocol = Path(protocol)
    output = Path(output_dir) if output_dir is not None else protocol / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    methods = [('dagger', 'Generated: 9 episodes'), ('dagger_offline', 'Loaded offline data')]
    colors = ['tab:blue', 'tab:orange']
    metrics = [('completion_rate', 'Completion rate'), ('steps_mean', 'Steps - all evaluation episodes'), ('finish_steps_mean', 'Steps - completed episodes')]
    for seed in seeds:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for (method, label), color in zip(methods, colors):
            rows = pd.read_csv(protocol / f'{method}_seed{seed}' / 'evaluation.csv').sort_values('online_steps')
            for ax, (metric, ylabel) in zip(axes, metrics):
                ax.plot(rows['online_steps'], rows[metric], color=color, marker='.', label=label)
                std = {'steps_mean': 'steps_std', 'finish_steps_mean': 'finish_steps_std'}.get(metric)
                if std is not None:
                    x = rows['online_steps'].to_numpy()
                    y = rows[metric].to_numpy(dtype=float)
                    spread = rows[std].to_numpy(dtype=float)
                    ax.fill_between(x, y - spread, y + spread, color=color, alpha=0.15)
                ax.set(xlabel='Online collection steps', ylabel=ylabel)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        axes[0].set_ylim(-0.05, 1.05)
        fig.suptitle(f'Initialization comparison - training seed {seed}; bands: evaluation episode std')
        fig.tight_layout()
        path = output / f'initialization_seed{seed}.svg'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(path)
    winners = []
    rows_out = []
    for (method, label), color in zip(methods, colors):
        candidates = []
        for seed in seeds:
            folder = protocol / f'{method}_seed{seed}'
            best = json.loads((folder / 'best_evaluation.json').read_text())
            candidates.append((seed, folder, best))
        seed, folder, best = min(candidates, key=lambda item: (-item[2]['completion_rate'], item[2]['finish_steps_mean'] if item[2]['finish_steps_mean'] is not None else np.inf, item[0]))
        final_path = folder / 'final_test_episodes.csv'
        if not final_path.exists():
            print(f'Best-actor final comparison skipped: run final_test.py on {folder} --checkpoints best --runs 20')
            return
        episodes = pd.read_csv(final_path)
        episodes = episodes[episodes['checkpoint'] == 'best'].sort_values('seed')
        if len(episodes) != 20 or episodes['seed'].nunique() != 20:
            raise ValueError(f'Expected 20 distinct final-test seeds for {folder}')
        completed = episodes['terminated'].astype(str).str.lower().eq('true')
        successful = episodes.loc[completed, 'n_steps']
        summary = dict(completion_rate=float(completed.mean()), steps_mean=float(episodes['n_steps'].mean()), steps_std=float(episodes['n_steps'].std()), finish_steps_mean=float(successful.mean()) if len(successful) else None, finish_steps_std=float(successful.std()) if len(successful) > 1 else None)
        winners.append((label, color, seed, best, episodes))
        rows_out.append(dict(method=method, training_seed=seed, source=str(folder), selected_iteration=best['iteration'], **summary))
    reference_seeds = winners[0][-1]['seed'].tolist()
    if any((w[-1]['seed'].tolist() != reference_seeds for w in winners)):
        raise ValueError('Best actors need matching evaluation seeds for paired comparison')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(reference_seeds))
    for label, color, seed, best, episodes in winners:
        completed = episodes['terminated'].astype(str).str.lower().eq('true').astype(int)
        axes[0].plot(x, completed, marker='o', color=color, label=f'{label}: {completed.mean():.0%}')
        axes[1].plot(x, episodes['n_steps'], marker='o', color=color, label=f"{label} (training seed {seed}, iteration {best['iteration']})")
        mean_steps = episodes['n_steps'].mean()
        std_steps = episodes['n_steps'].std(ddof=1)
        axes[1].axhline(mean_steps, color=color, linestyle='--', alpha=0.8, label=f'{label}: mean {mean_steps:.1f} ± std {std_steps:.1f}')
        axes[1].axhspan(mean_steps - std_steps, mean_steps + std_steps, color=color, alpha=0.12)
    for ax in axes:
        ax.set_xticks(x, [str(s) for s in reference_seeds], rotation=45)
        ax.set_xlabel('Evaluation seed (paired episodes)')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set(ylabel='Completed lap: 1=yes, 0=no', ylim=(-0.05, 1.05))
    axes[1].set_ylabel('Episode steps (timeouts included)')
    fig.suptitle('Selected best actors - 20 held-out seeds; dashed mean, shaded ±1 episode std')
    fig.tight_layout()
    path = output / 'initialization_best_actors.svg'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows_out).to_csv(output / 'initialization_best_actors.csv', index=False)
    print(path)
learning_ROOT = Path(__file__).resolve().parents[2]
learning_COLORS = {'Behavioral Cloning': '#55A868', 'DAgger': '#4C72B0', 'RLPD': '#DD8452'}

def learning_load_history(protocol, seeds, interval=5000, total=50000):
    tables = []
    for method, prefix in [('DAgger', 'dagger_offline_gaussian'), ('RLPD', 'rlpd')]:
        for seed in seeds:
            directory = protocol / f'{prefix}_seed{seed}'
            data = pd.read_csv(directory / 'evaluation.csv')
            data['actual_online_steps'] = data.online_steps if method == 'DAgger' else data.env_steps
            data['target_online_steps'] = ((data.iteration - 1) * interval if method == 'DAgger' else data.env_steps // interval * interval).astype(int)
            if data.target_online_steps.duplicated().any():
                raise ValueError(f'Duplicate evaluation targets at {directory}')
            expected = list(range(0 if method == 'DAgger' else interval, total + 1, interval))
            if set(data.target_online_steps) - set(expected):
                raise ValueError(f'Unexpected evaluation targets at {directory}')
            data.loc[data.completed.eq(0), 'finish_steps_mean'] = np.nan
            data = data.set_index('target_online_steps').reindex(expected).rename_axis('target_online_steps').reset_index()
            data['method'] = method
            data['training_seed'] = seed
            tables.append(data)
    return pd.concat(tables, ignore_index=True)

def learning_plot(protocol, output, seeds, metrics=None):
    history = learning_load_history(protocol, seeds)
    output.mkdir(parents=True, exist_ok=True)
    history.to_csv(output / 'singlemap_learning_history.csv', index=False)
    initial = history[(history.method == 'DAgger') & history.target_online_steps.eq(0)]
    if len(initial) != len(seeds):
        raise ValueError('Missing BC baseline')
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 11})
    summaries = []
    for metric, title, ylabel, filename, multiplier in [('completion_rate', 'Single-map completion during training', 'Completion rate (%)', 'singlemap_completion_evolution.svg', 100), ('finish_steps_mean', 'Single-map finishing steps - mean ± std across training seeds', 'Steps to finish', 'singlemap_finishing_steps_evolution.svg', 1)]:
        fig, ax = plt.subplots(figsize=(8.5, 3.2) if metric == 'finish_steps_mean' else (11, 4.3))
        if metrics is not None and ('completion' if metric == 'completion_rate' else 'finishing-steps') not in metrics:
            continue
        baseline = initial[metric]
        if baseline.notna().all():
            mean = baseline.mean() * multiplier
            std = baseline.std(ddof=1) * multiplier
            ax.axhline(mean, color=learning_COLORS['Behavioral Cloning'], linestyle='--', linewidth=1.7, label='Behavioral Cloning')
            low, high = (mean - std, mean + std)
            if metric == 'completion_rate':
                low, high = (max(0, low), min(100, high))
            ax.axhspan(low, high, color=learning_COLORS['Behavioral Cloning'], alpha=0.1)
        for method in ('DAgger', 'RLPD'):
            rows = history[history.method == method]
            grouped = rows.groupby('target_online_steps')[metric].agg(['mean', 'std', 'count']).reset_index()
            grouped.loc[grouped['count'] != len(seeds), ['mean', 'std']] = np.nan
            x = grouped.target_online_steps.to_numpy()
            mean = grouped['mean'].to_numpy() * multiplier
            std = grouped['std'].to_numpy() * multiplier
            ax.plot(x, mean, color=learning_COLORS[method], marker='o' if method == 'DAgger' else '^', markersize=7 if method == 'DAgger' else 4, markerfacecolor='none' if method == 'DAgger' else learning_COLORS[method], linestyle='-' if method == 'DAgger' else '-.', linewidth=1.6, label=method)
            low, high = (mean - std, mean + std)
            if metric == 'completion_rate':
                low = np.maximum(0, low)
                high = np.minimum(100, high)
            ax.fill_between(x, low, high, color=learning_COLORS[method], alpha=0.17)
            grouped['method'] = method
            grouped['metric'] = metric
            summaries.append(grouped)
        if metric == 'finish_steps_mean':
            reference = pd.read_csv(protocol / f'dagger_offline_gaussian_seed{seeds[0]}' / 'expert_benchmark/summary.csv')
            expert_mean = float(reference.loc[reference.track == 'abyss', 'finish_steps_mean'].iloc[0])
            ax.axhline(expert_mean, color='#333333', linestyle=':', linewidth=1.6, label=f'Expert (team4): {expert_mean:.1f} steps')
        if metric == 'completion_rate':
            ax.set_ylim(-3, 103)
        ax.set(title=title, xlabel='Target online steps', ylabel=ylabel, xlim=(0, 50000))
        ax.set_xticks(np.arange(0, 50001, 5000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:g}k'))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4 if metric == 'finish_steps_mean' else 3, frameon=False, fontsize=9 if metric == 'finish_steps_mean' else 11)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        fig.savefig(output / filename, bbox_inches='tight')
        plt.close(fig)
        print(output / filename)
    pd.concat(summaries, ignore_index=True).to_csv(output / 'singlemap_learning_summary.csv', index=False)
    initial[['training_seed', 'episodes', 'completed', 'completion_rate', 'finish_steps_mean']].to_csv(output / 'singlemap_bc_learning_baseline.csv', index=False)
    notes = 'Source: single-map Abyss Gaussian DAgger and RLPD runs, training seeds ' + str(seeds) + '.\nEvery recorded evaluation is used, not a running best checkpoint.\nTargets: DAgger (iteration-1)*5000; RLPD floor(actual online steps/5000)*5000. Actual counts are preserved.\nMissing targets remain missing. RLPD has no invented zero-step value.\nBC is the initial offline-only DAgger validation baseline, held constant.\nMeans and sample standard deviations are across training-seed evaluation means, not individual laps or confidence intervals.\nFinishing steps use successful laps only. A combined point is omitted if any training seed has no completed lap.\n'
    (output / 'singlemap_learning_methodology.txt').write_text(notes)
    assert history.groupby(['method', 'training_seed']).episodes.first().eq(5).all()
    return history
bc_ROOT = Path(__file__).resolve().parents[2]

def bc_plot(args):
    expert = final_load_results(args.expert, checkpoint='team4')
    seeds = expert.index
    if len(seeds) != 20 or not expert.terminated.all():
        raise ValueError('Require 20 completed expert tests')
    training_seeds = getattr(args, 'training_seeds', None)
    if training_seeds:
        if len(set(training_seeds)) != len(training_seeds):
            raise ValueError('Duplicate training seeds')
        data = {}
        for name, prefix, checkpoint in [('BC', 'bc_gaussian', 'bc'), ('DAgger', 'dagger_offline_gaussian', 'best'), ('RLPD', 'rlpd', 'best')]:
            frames = []
            for seed in training_seeds:
                rows = final_load_results(args.protocol_dir / f'{prefix}_seed{seed}', checkpoint=checkpoint)
                if not rows.index.equals(seeds):
                    raise ValueError(f'Mismatched race seeds: {name}, training seed {seed}')
                frames.append(rows)
            data[name] = pd.concat(frames, keys=training_seeds, names=['training_seed', 'seed'])
    else:
        data = {'BC': final_load_results(args.bc_results, checkpoint='bc'), 'DAgger': final_load_results(args.dagger), 'RLPD': final_load_results(args.rlpd)}
        if not all((rows.index.equals(seeds) for rows in data.values())):
            raise ValueError('Test seeds must match')
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 11, 'axes.titlesize': 15, 'axes.labelsize': 12})
    fig, ax = plt.subplots(figsize=(6, 5.5))
    distributions = [rows.n_steps.to_numpy(dtype=float) for rows in data.values()]
    box = ax.boxplot(distributions, tick_labels=['Behavioral Cloning' if name == 'BC' else name for name in data], patch_artist=True, showfliers=False, widths=0.28, medianprops=dict(color='#222222', linewidth=2), whiskerprops=dict(color='#333333', linewidth=1.5), capprops=dict(color='#333333', linewidth=1.5))
    colors = ['#55A868', '#4C72B0', '#DD8452']
    summaries = []
    for index, ((name, rows), steps, color, patch) in enumerate(zip(data.items(), distributions, colors, box['boxes']), start=1):
        patch.set(facecolor=color, edgecolor=color, alpha=0.6)
        finished = rows.terminated.to_numpy()
        if training_seeds:
            from matplotlib.lines import Line2D
            markers = ['o', 's', '^', 'D', 'v', 'P']
            if len(training_seeds) > len(markers):
                raise ValueError('At most six training seeds supported')
            for group, (seed, center) in enumerate(zip(training_seeds, np.linspace(-0.1, 0.1, len(training_seeds)))):
                mask = rows.index.get_level_values('training_seed') == seed
                values = steps[mask]
                success = finished[mask]
                offsets = center + np.linspace(-0.025, 0.025, len(values))
                ax.scatter(index + offsets[success], values[success], marker=markers[group], color=color, edgecolor='white', linewidth=0.4, alpha=0.55, s=20, zorder=3)
                mean_color = tuple(np.asarray(matplotlib.colors.to_rgb(color)) * 0.55)
                ax.scatter(index + center, values.mean(), marker=markers[group], color=mean_color, edgecolor='black', linewidth=0.8, s=60, alpha=0.6, zorder=5)
                if (~success).any():
                    ax.scatter(index + offsets[~success], values[~success], marker='x', color=color, s=28, zorder=3)
        else:
            offsets = np.linspace(-0.1, 0.1, len(seeds))
            for mask, marker in ((finished, 'o'), (~finished, 'x')):
                ax.scatter(index + offsets[mask], steps[mask], marker=marker, color=color, edgecolor='white' if marker == 'o' else None, linewidth=0.5 if marker == 'o' else 1.2, alpha=0.9, s=34, zorder=3)
        finish = steps[finished]
        summaries.append(dict(method=name, episodes=len(steps), completed=int(finished.sum()), completion_rate=float(finished.mean()), steps_mean=steps.mean(), steps_std=steps.std(ddof=1), finish_steps_mean=finish.mean() if len(finish) else None, finish_steps_std=finish.std(ddof=1) if len(finish) > 1 else None))
    expert_mean = float(expert.n_steps.mean())
    ax.axhline(expert_mean, color='#333333', linestyle='--', linewidth=1.4, label=f'Expert (team4) mean: {expert_mean:.2f} steps')
    ax.legend(frameon=False, fontsize=10)
    title = f'Distribution of episode steps\n{len(training_seeds)} training seeds × {len(seeds)} test seeds' if training_seeds else f'Distribution of episode steps across {len(seeds)} test seeds'
    ax.set(title=title, ylabel='Episode steps')
    if training_seeds:
        handles = [Line2D([], [], color='gray', marker=markers[i], linestyle='', label=str(seed)) for i, seed in enumerate(training_seeds)]
        fig.legend(handles=handles, title='Training seed (large outlined markers = means)', loc='lower center', ncol=len(training_seeds), frameon=False, fontsize=9, title_fontsize=9)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
    else:
        fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / 'bc_dagger_rlpd_step_distribution.svg'
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(args.output_dir / 'bc_dagger_rlpd_final_summary.csv', index=False)
    paired = pd.DataFrame({name.lower() + '_steps': rows.n_steps for name, rows in data.items()})
    paired.reset_index().to_csv(args.output_dir / 'bc_dagger_rlpd_final_paired.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(target)

def comparison_plot_results(output, destination):
    output = Path(output)
    rows = pd.read_csv(output / 'episodes.csv')
    summary = pd.read_csv(output / 'summary.csv')
    rows['terminated'] = rows['terminated'].astype(str).str.lower().eq('true')
    tracks = list(dict.fromkeys(rows['track']))
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['svg.fonttype'] = 'none'
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    expert = pd.read_csv(output / 'expert/summary.csv').set_index('track')
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(tracks), figsize=(19, 4.5), squeeze=False)
    labels = {'abyss': 'Abyss', 'volcano_island': 'Volcan Island', 'sandtrack': 'Shifting Sands', 'snowtuxpeak': 'Snow Peak', 'xr591': 'XR591', 'zengarden': 'Zen Garden'}
    colors = {'DAgger': 'tab:blue', 'RLPD': 'tab:orange'}
    for ax, track in zip(axes[0], tracks):
        for position, (model, color) in enumerate(colors.items()):
            episode = rows[(rows['model'] == model) & (rows['track'] == track)].sort_values('seed')
            box = ax.boxplot([episode['n_steps']], positions=[position], widths=0.35, patch_artist=True, showfliers=False, manage_ticks=False, medianprops=dict(color=color, linewidth=1.8))
            box['boxes'][0].set(facecolor=color, alpha=0.3)
            offsets = np.linspace(-0.07, 0.07, len(episode))
            finished = episode['terminated'].to_numpy()
            for mask, marker in [(finished, 'o'), (~finished, 'x')]:
                ax.scatter(position + offsets[mask], episode['n_steps'].to_numpy()[mask], color=color, marker=marker, s=20, zorder=3, alpha=0.75)
            ax.text(position, 1.02, f'{finished.sum()}/{len(episode)} finish', transform=ax.get_xaxis_transform(), ha='center', fontsize=9, color=color)
        ax.axhline(expert.loc[track, 'finish_steps_mean'], color='#333333', linestyle='--', linewidth=1.3)
        ax.set_xticks([0, 1], ['DAgger', 'RLPD'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(labels.get(track, track), pad=30)
        ax.set_ylabel('Episode steps')
        ax.margins(y=0.15)
    handles = [Patch(facecolor=c, alpha=0.5, label=m) for m, c in colors.items()]
    handles += [Line2D([], [], color='#333333', linestyle='--', label='Expert (team4) mean'), Line2D([], [], color='gray', marker='o', linestyle='', label='Completed lap'), Line2D([], [], color='gray', marker='x', linestyle='', label='Incomplete lap (timeout)')]
    fig.suptitle('DAgger and RLPD across maps - 20 evaluation seeds', y=0.99)
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    path = output / 'generalization_boxplots.svg'
    fig.savefig(path, bbox_inches='tight')
    print(path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Plot saved results only - never run simulator evaluations.')
    parser.add_argument('--setup', choices=['singlemap', 'multimap'], required=True)
    parser.add_argument('--plot', choices=['all', 'learning', 'distribution', 'completion', 'relative', 'finishing-steps', 'paired', 'run', 'initialization', 'generalization'], default='all')
    parser.add_argument('--protocol-dir', type=Path, default=bc_ROOT / 'datasets/protocol')
    parser.add_argument('--root', type=Path, default=bc_ROOT / 'datasets/multimap/full5maps50k')
    parser.add_argument('--output-dir', type=Path, default=bc_ROOT / 'datasets/plots')
    parser.add_argument('--training-seeds', type=int, nargs='+', default=[1000, 2000, 3000])
    parser.add_argument('--comparison-dir', type=Path)
    parser.add_argument('--experiment', type=Path, help='Run folder for --plot run')
    parser.add_argument('--dagger', type=Path)
    parser.add_argument('--rlpd', type=Path)
    parser.add_argument('--expert', type=Path)
    args = parser.parse_args()
    if len(args.training_seeds) != 3 or len(set(args.training_seeds)) != 3:
        parser.error('Require three distinct training seeds for the report comparisons')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.setup == 'multimap':
        from utils.multimap_reports import read_runs, aligned_history, plot_evolution, make_jobs, final_tables, plot_final
        if args.plot not in ('all', 'learning', 'completion', 'distribution', 'relative'):
            parser.error('Multimap supports all, learning, completion, distribution or relative')
        runs, reference = read_runs(args.root, args.training_seeds)
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
        if args.plot in ('all', 'learning'):
            history = aligned_history(runs)
            history.to_csv(args.output_dir / 'multimap_evaluation_history.csv', index=False)
            plot_evolution(history, reference, args.output_dir, args.training_seeds)
        if args.plot != 'learning':
            episodes, summary = final_tables(make_jobs(args.root, runs, reference, args.training_seeds), reference)
            plot_final(episodes, summary, reference, args.output_dir, args.training_seeds, plots=None if args.plot == 'all' else {args.plot})
        return
    if args.plot == 'generalization':
        if args.comparison_dir is None: parser.error('--plot generalization requires --comparison-dir')
        comparison_plot_results(args.comparison_dir, args.output_dir)
        return
    protocol = args.protocol_dir
    if args.plot in ('all', 'learning', 'completion', 'finishing-steps'):
        learning_plot(protocol, args.output_dir, args.training_seeds, metrics=None if args.plot in ('all', 'learning') else {args.plot})
    if args.plot in ('all', 'distribution'):
        args.expert = args.expert or protocol / 'expert_test'
        bc_plot(args)
    if args.plot in ('all', 'relative'):
        dagger_plot_relative_comparison(protocol, args.training_seeds, args.output_dir)
    if args.plot == 'finishing-steps':
        return
    if args.plot == 'paired':
        if args.dagger is None or args.rlpd is None:
            parser.error('--plot paired requires --dagger and --rlpd run folders')
        final_plot_comparison(args.dagger, args.rlpd, args.output_dir, 'DAgger', 'RLPD', args.expert)
    if args.plot == 'run':
        if args.experiment is None:
            parser.error('--plot run requires --experiment')
        dagger_plot_results(args.experiment, args.output_dir)
    if args.plot == 'initialization':
        dagger_plot_initialization_details(protocol, args.training_seeds, args.output_dir)
if __name__ == '__main__':
    main()
