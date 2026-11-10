"""Compare selected DAgger/RLPD best actors on matched held-out race seeds."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(directory):
    rows = pd.read_csv(Path(directory) / 'final_test_episodes.csv')
    rows = rows.loc[rows['checkpoint'] == 'best'].copy().sort_values('seed')
    if rows.empty or rows['seed'].duplicated().any():
        raise ValueError(f'Missing or duplicate best-checkpoint test seeds: {directory}')
    for key in ('terminated', 'truncated'):
        rows[key] = rows[key].astype(str).str.lower().eq('true')
    return rows.set_index('seed')


def plot_comparison(dagger, rlpd, output_dir, dagger_label='DAgger (loaded offline)', rlpd_label='RLPD'):
    data = {dagger_label: load_results(dagger), rlpd_label: load_results(rlpd)}
    left, right = data.values()
    if not left.index.equals(right.index):
        raise ValueError('Both actors must have exactly the same final-test seeds')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = left.index.to_numpy()
    x = np.arange(len(seeds))
    # Seaborn darkgrid appearance bundled with Matplotlib; no extra dependency.
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 15,
                         'axes.labelsize': 12, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = ['#4C72B0', '#DD8452']
    summaries, distributions = [], []
    fig, ax = plt.subplots(figsize=(15, 4.8))
    for (name, rows), color in zip(data.items(), colors):
        steps = rows['n_steps'].to_numpy(dtype=float)
        completed = rows['terminated'].to_numpy()
        mean, std = steps.mean(), steps.std(ddof=1) if len(steps) > 1 else np.nan
        ax.plot(x, steps, marker='o', markersize=5, color=color, linewidth=1.6,
                label=f'{"DAgger" if name == dagger_label else "RLPD"}: {mean:.2f} ± {std:.2f} steps')
        ax.scatter(x[~completed], steps[~completed], color=color, marker='x', s=100)
        ax.axhline(mean, color=color, linestyle='--', linewidth=1.2)
        ax.axhspan(mean - std, mean + std, color=color, alpha=.12)
        distributions.append(steps)
        finish = steps[completed]
        summaries.append(dict(method=name, episodes=len(steps), completed=int(completed.sum()),
            completion_rate=float(completed.mean()), steps_mean=mean, steps_std=std,
            finish_steps_mean=finish.mean() if len(finish) else np.nan,
            finish_steps_std=finish.std(ddof=1) if len(finish) > 1 else np.nan))
    ax.set(title=f'Steps to finish across {len(seeds)} test seeds - mean ± standard deviation',
           xlabel='Test seed', ylabel='Episode steps')
    ax.set_xticks(x, [str(seed) for seed in seeds], rotation=45, ha='right')
    all_steps = np.concatenate(distributions)
    ax.set_ylim(all_steps.min() - 10, all_steps.max() + 10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045),
               ncol=2, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    for suffix in ('pdf', 'svg'):
        path = output / f'dagger_rlpd_paired_episodes.{suffix}'
        fig.savefig(path, bbox_inches='tight')
        print(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    box = ax.boxplot(distributions, tick_labels=list(data), patch_artist=True,
                     showfliers=False, widths=.28,
                     medianprops=dict(color='#222222', linewidth=2),
                     whiskerprops=dict(color='#333333', linewidth=1.5),
                     capprops=dict(color='#333333', linewidth=1.5))
    offsets = np.linspace(-.10, .10, len(seeds))
    for index, (steps, color, patch) in enumerate(zip(distributions, colors, box['boxes']), start=1):
        patch.set(facecolor=color, edgecolor=color, alpha=.6)
        ax.scatter(index + offsets, steps, color=color, edgecolor='white',
                   linewidth=.5, alpha=.9, s=34, zorder=3)
    ax.set(title=f'Distribution of episode steps across {len(seeds)} test seeds',
           ylabel='Episode steps')
    fig.tight_layout()
    for suffix in ('pdf', 'svg'):
        path = output / f'dagger_rlpd_step_distribution.{suffix}'
        fig.savefig(path, bbox_inches='tight')
        print(path)
    plt.close(fig)
    both = left['terminated'] & right['terminated']
    values = (right['n_steps'] - left['n_steps']).where(both).to_numpy(dtype=float)
    pd.DataFrame(summaries).to_csv(output / 'dagger_rlpd_final_summary.csv', index=False)
    pd.DataFrame(dict(seed=seeds, dagger_steps=left['n_steps'].to_numpy(),
        rlpd_steps=right['n_steps'].to_numpy(), both_completed=both.to_numpy(),
        rlpd_minus_dagger_steps=values)).to_csv(output / 'dagger_rlpd_final_paired.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dagger', type=Path, required=True, help='DAgger run folder with final_test_episodes.csv')
    parser.add_argument('--rlpd', type=Path, required=True, help='RLPD run folder with final_test_episodes.csv')
    parser.add_argument('--output-dir', type=Path, default=Path('datasets/plots'))
    parser.add_argument('--dagger-label', default='DAgger (loaded offline)')
    parser.add_argument('--rlpd-label', default='RLPD')
    args = parser.parse_args()
    plot_comparison(args.dagger, args.rlpd, args.output_dir, args.dagger_label, args.rlpd_label)
