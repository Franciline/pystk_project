"""Plot DAgger CSV logs, including per-map expert baselines."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(experiment, output_dir=None):
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
                    ax.axvline(start + .5, color='grey', alpha=.3, linewidth=1)
                ax.set(title=column.upper(), xlabel='Logged training epoch', ylabel='Loss')
                ax.grid(alpha=.2)
            fig.suptitle('Training loss — vertical lines separate DAgger iterations')
            save(fig, 'losses.svg')

    evaluation = experiment / 'evaluation.csv'
    if not evaluation.exists():
        print('No evaluation.csv yet; evaluation plots skipped.')
        return saved
    data = pd.read_csv(evaluation)
    if data.empty:
        return saved
    # Older single-map logs did not include a track column.
    if 'track' not in data:
        data['track'] = 'training_track'
    baseline_path = experiment / 'expert_benchmark' / 'summary.csv'
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    tracks = list(dict.fromkeys(data['track']))
    fig, axes = plt.subplots(3, len(tracks), figsize=(4.5 * len(tracks), 10), squeeze=False)
    for col, track in enumerate(tracks):
        rows = data[data['track'] == track].sort_values('iteration')
        expert = baseline[baseline['track'] == track] if not baseline.empty else baseline
        diagnostic = 'selection_track' in rows and not rows['selection_track'].astype(str).str.lower().eq('true').any()
        axes[0, col].set_title(f'{track}' + (' (unseen)' if diagnostic else ''))
        for index, (metric, std, label) in enumerate([
            ('completion_rate', None, 'Completion rate'),
            ('steps_mean', 'steps_std', 'Steps — all episodes'),
            ('finish_steps_mean', 'finish_steps_std', 'Steps — completed episodes'),
        ]):
            ax = axes[index, col]
            x = rows['iteration'].to_numpy()
            y = rows[metric].to_numpy(dtype=float)
            ax.plot(x, y, marker='o', label='Actor')
            if std in rows:
                spread = rows[std].to_numpy(dtype=float)
                ax.fill_between(x, y - spread, y + spread, alpha=.2)
            if not expert.empty and pd.notna(expert.iloc[0][metric]):
                level = float(expert.iloc[0][metric])
                ax.axhline(level, color='tab:orange', linestyle='--', label='Expert')
                if std in expert and pd.notna(expert.iloc[0][std]):
                    spread = float(expert.iloc[0][std])
                    ax.axhspan(level - spread, level + spread, color='tab:orange', alpha=.12)
            ax.set(xlabel='DAgger iteration', ylabel=label)
            if metric == 'completion_rate':
                ax.set_ylim(-.05, 1.05)
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Actor versus expert — bands show episode standard deviation, not confidence intervals')
    save(fig, 'evaluation_by_track.svg')

    if 'relative_finish_steps' in data:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for track in tracks:
            rows = data[data['track'] == track].sort_values('iteration')
            ax.plot(rows['iteration'], rows['relative_finish_steps'], marker='o', label=track)
        ax.axhline(1, linestyle='--', color='black', label='Expert reference')
        ax.set(xlabel='DAgger iteration', ylabel='Actor / expert mean finishing steps',
               title='Relative speed — completed episodes only; lower is faster')
        ax.grid(alpha=.2)
        ax.legend()
        save(fig, 'relative_steps.svg')

    aggregate_path = experiment / 'evaluation_aggregate.csv'
    if aggregate_path.exists():
        rows = pd.read_csv(aggregate_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, label in zip(axes, ['completion_rate', 'relative_finish_steps'],
                                    ['Mean training-map completion rate', 'Mean training-map step ratio']):
            ax.plot(rows['training_steps'], rows[metric], marker='o')
            ax.set(xlabel='Cumulative collection steps', ylabel=label)
            ax.grid(alpha=.2)
        save(fig, 'training_maps_aggregate.svg')
    return saved


def plot_relative_comparison(protocol, seeds, output_dir=None, relative=True):
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
            candidates.append(((-best['completion_rate'],
                best['finish_steps_mean'] if best['finish_steps_mean'] is not None else np.inf,
                seed), folder))
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
    for method, label, color in [('dagger_offline_gaussian','DAgger','#4C72B0'),
                                  ('rlpd','RLPD','#DD8452')]:
        folder = selected[method]
        settings = json.loads((folder / 'config.json').read_text())
        if any(settings[k] != config[k] for k in ('track','step_size','max_steps')) or settings['evaluation']['seeds'] != config['evaluation']['seeds']:
            raise ValueError('Comparison requires matching task and evaluation seeds')
        rows = pd.read_csv(folder / 'evaluation.csv')
        column = 'online_steps' if 'online_steps' in rows else 'env_steps'
        rows = rows.sort_values(column)
        x = rows[column].to_numpy(dtype=float)
        y = rows['finish_steps_mean'].to_numpy(dtype=float) / scale
        std = rows['finish_steps_std'].to_numpy(dtype=float) / scale
        ax.plot(x, y, marker='o', markersize=4, color=color, label=label)
        ax.fill_between(x, y - std, y + std, color=color, alpha=.15)
        for i, (_, row) in enumerate(rows.iterrows()):
            summaries.append(dict(method=label, source=str(folder), online_steps=x[i],
                mean_steps=row['finish_steps_mean'], std_steps=row['finish_steps_std'],
                relative_steps=row['finish_steps_mean'] / reference,
                relative_std=row['finish_steps_std'] / reference, expert_mean_steps=reference,
                completion_rate=row['completion_rate']))
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


def plot_initialization_details(protocol, seeds, output_dir=None):
    """All training-seed histories, then paired evaluation of selected best actors."""
    protocol = Path(protocol)
    output = Path(output_dir) if output_dir is not None else protocol / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    methods = [('dagger', 'Generated: 9 episodes'),
               ('dagger_offline', 'Loaded offline data')]
    colors = ['tab:blue', 'tab:orange']
    metrics = [('completion_rate', 'Completion rate'),
               ('steps_mean', 'Steps — all evaluation episodes'),
               ('finish_steps_mean', 'Steps — completed episodes')]
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
                    ax.fill_between(x, y - spread, y + spread, color=color, alpha=.15)
                ax.set(xlabel='Online collection steps', ylabel=ylabel)
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        axes[0].set_ylim(-.05, 1.05)
        fig.suptitle(f'Initialization comparison — training seed {seed}; bands: evaluation episode std')
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
        seed, folder, best = min(candidates, key=lambda item: (
            -item[2]['completion_rate'],
            item[2]['finish_steps_mean'] if item[2]['finish_steps_mean'] is not None else np.inf,
            item[0]))
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
        summary = dict(completion_rate=float(completed.mean()),
                       steps_mean=float(episodes['n_steps'].mean()),
                       steps_std=float(episodes['n_steps'].std()),
                       finish_steps_mean=float(successful.mean()) if len(successful) else None,
                       finish_steps_std=float(successful.std()) if len(successful) > 1 else None)
        winners.append((label, color, seed, best, episodes))
        rows_out.append(dict(method=method, training_seed=seed, source=str(folder),
                             selected_iteration=best['iteration'], **summary))
    reference_seeds = winners[0][-1]['seed'].tolist()
    if any(w[-1]['seed'].tolist() != reference_seeds for w in winners):
        raise ValueError('Best actors need matching evaluation seeds for paired comparison')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(reference_seeds))
    for label, color, seed, best, episodes in winners:
        completed = episodes['terminated'].astype(str).str.lower().eq('true').astype(int)
        axes[0].plot(x, completed, marker='o', color=color,
                     label=f'{label}: {completed.mean():.0%}')
        axes[1].plot(x, episodes['n_steps'], marker='o', color=color,
                     label=f'{label} (training seed {seed}, iteration {best["iteration"]})')
        mean_steps = episodes['n_steps'].mean()
        std_steps = episodes['n_steps'].std(ddof=1)
        axes[1].axhline(mean_steps, color=color, linestyle='--', alpha=.8,
                        label=f'{label}: mean {mean_steps:.1f} ± std {std_steps:.1f}')
        axes[1].axhspan(mean_steps - std_steps, mean_steps + std_steps,
                       color=color, alpha=.12)
    for ax in axes:
        ax.set_xticks(x, [str(s) for s in reference_seeds], rotation=45)
        ax.set_xlabel('Evaluation seed (paired episodes)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set(ylabel='Completed lap: 1=yes, 0=no', ylim=(-.05, 1.05))
    axes[1].set_ylabel('Episode steps (timeouts included)')
    fig.suptitle('Selected best actors — 20 held-out seeds; dashed mean, shaded ±1 episode std')
    fig.tight_layout()
    path = output / 'initialization_best_actors.svg'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows_out).to_csv(output / 'initialization_best_actors.csv', index=False)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--experiment', type=Path, help='Directory containing evaluation.csv and losses.csv')
    mode.add_argument('--steps-best', type=Path, help='Actual finishing steps for best Gaussian DAgger and RLPD runs')
    mode.add_argument('--relative-best', type=Path, help='Relative curves for best Gaussian DAgger and RLPD runs')
    mode.add_argument('--compare-protocol', type=Path, help='Compare dagger_seed* and dagger_offline_seed* runs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1000, 2000, 3000])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.steps_best is not None:
        plot_relative_comparison(args.steps_best, args.seeds, args.output_dir, relative=False)
    elif args.relative_best is not None:
        plot_relative_comparison(args.relative_best, args.seeds, args.output_dir)
    elif args.compare_protocol is not None:
        plot_relative_comparison(args.compare_protocol, args.seeds, args.output_dir)
        plot_initialization_details(args.compare_protocol, args.seeds, args.output_dir)
    else:
        plot_results(args.experiment, args.output_dir)
