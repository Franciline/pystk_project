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
            save(fig, 'losses.png')

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
    save(fig, 'evaluation_by_track.png')

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
        save(fig, 'relative_steps.png')

    aggregate_path = experiment / 'evaluation_aggregate.csv'
    if aggregate_path.exists():
        rows = pd.read_csv(aggregate_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, label in zip(axes, ['completion_rate', 'relative_finish_steps'],
                                    ['Mean training-map completion rate', 'Mean training-map step ratio']):
            ax.plot(rows['training_steps'], rows[metric], marker='o')
            ax.set(xlabel='Cumulative collection steps', ylabel=label)
            ax.grid(alpha=.2)
        save(fig, 'training_maps_aggregate.png')
    return saved


def plot_relative_comparison(protocol, seeds, output_dir=None):
    """Compare seed means on a shared online-step grid; carry latest eval forward.

    Bands are sample std across training seeds, not across evaluation episodes.
    Each run uses its own saved expert mean as a fixed denominator.
    """
    protocol = Path(protocol)
    output = Path(output_dir) if output_dir is not None else protocol / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    methods = {'dagger': 'DAgger — generated initialization',
               'dagger_offline': 'DAgger — loaded offline initialization'}
    runs = {}
    task = None
    for method in methods:
        runs[method] = []
        for seed in seeds:
            folder = protocol / f'{method}_seed{seed}'
            config = json.loads((folder / 'config.json').read_text())
            current = (config['track'], config['max_steps'], config['step_size'],
                       tuple(config['evaluation']['seeds']))
            if task is not None and current != task:
                raise ValueError('Comparison requires matching task and evaluation conditions')
            task = current
            rows = pd.read_csv(folder / 'evaluation.csv').sort_values('online_steps')
            expert = pd.read_csv(folder / 'expert_benchmark' / 'summary.csv')
            reference = float(expert.loc[expert['track'] == config['track'], 'finish_steps_mean'].iloc[0])
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError(f'No successful expert baseline: {folder}')
            rows['ratio'] = rows['finish_steps_mean'] / reference
            runs[method].append(rows)
    end = min(float(r['online_steps'].max()) for rows in runs.values() for r in rows)
    grid = np.unique(np.r_[np.arange(0, end + 1, 5000), end])
    fig, ax = plt.subplots(figsize=(9, 5))
    summaries = []
    for method, label in methods.items():
        values, completion = [], []
        for rows in runs[method]:
            indices = np.searchsorted(rows['online_steps'].to_numpy(), grid, side='right') - 1
            if np.any(indices < 0):
                raise ValueError('Each run needs an initialization evaluation at zero online steps')
            values.append(rows['ratio'].to_numpy()[indices])
            completion.append(rows['completion_rate'].to_numpy()[indices])
        values = np.stack(values)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.full(len(grid), np.nan)
        line, = ax.plot(grid, mean, marker='o', label=f'{label} (n={len(seeds)})')
        ax.fill_between(grid, mean - std, mean + std, color=line.get_color(), alpha=.18)
        for i, step in enumerate(grid):
            summaries.append(dict(method=method, online_steps=step, ratio_mean=mean[i],
                                  ratio_std=std[i], completion_rate_mean=np.mean(completion, axis=0)[i]))
    ax.axhline(1, color='black', linestyle='--', linewidth=1.3, label='Expert = 1')
    ax.set(xlabel='Online collection steps (initial data excluded)',
           ylabel='Actor / expert mean finishing steps',
           title=f'Relative finishing steps — {task[0]} (lower is faster)')
    ax.grid(alpha=.2)
    ax.legend()
    fig.text(.5, .015, 'Bands: ±1 sample std across training seeds. Latest evaluation carried forward. Successful laps only.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    path = output / 'relative_steps_comparison.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(output / 'relative_steps_comparison.csv', index=False)
    print(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--experiment', type=Path, help='Directory containing evaluation.csv and losses.csv')
    mode.add_argument('--compare-protocol', type=Path, help='Compare dagger_seed* and dagger_offline_seed* runs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1000, 2000, 3000])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.compare_protocol is not None:
        plot_relative_comparison(args.compare_protocol, args.seeds, args.output_dir)
    else:
        plot_results(args.experiment, args.output_dir)
