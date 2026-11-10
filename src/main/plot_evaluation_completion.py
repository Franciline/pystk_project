"""Compare completion at nominal 5k evaluation rounds; retain actual step counts."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_completion(dagger, rlpd, output):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    records = []
    for name, directory, column, initial, color, marker in [
        ('DAgger Gaussian', dagger, 'online_steps', True, 'tab:blue', 'o'),
        ('RLPD', rlpd, 'env_steps', False, 'tab:orange', 'x'),
    ]:
        rows = pd.read_csv(Path(directory) / 'evaluation.csv').sort_values(column)
        rounds = np.arange(len(rows)) + (0 if initial else 1)
        ax.plot(rounds, rows['completion_rate'], label=name, color=color,
                marker=marker, markersize=8, linestyle='-' if initial else '--')
        for round_id, (_, row) in zip(rounds, rows.iterrows()):
            records.append(dict(method=name, evaluation_round=int(round_id),
                nominal_online_steps=int(round_id * 5000), actual_online_steps=int(row[column]),
                completed=int(row['completed']), episodes=int(row['episodes']),
                completion_rate=float(row['completion_rate'])))
    last = max(r['evaluation_round'] for r in records)
    ax.set_xticks(range(last + 1), ['0 (post-offline)'] + [f'{i * 5}k' for i in range(1, last + 1)])
    ax.set(xlabel='Evaluation round (nominal online collection budget)',
           ylabel='Completion rate', ylim=(-.03, 1.08),
           title='Gaussian DAgger vs RLPD — training seed 1000')
    ax.grid(alpha=.2)
    ax.legend()
    fig.text(.5, .01, 'Five evaluation laps per point. Actual step counts differ because evaluation waits for episode boundaries.\nDAgger at 0 is after offline training; no RLPD evaluation was recorded at 0.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        path = output / f'gaussian_dagger_rlpd_completion.{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        print(path)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output / 'gaussian_dagger_rlpd_completion.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dagger', type=Path, default=Path('datasets/protocol/dagger_offline_gaussian_seed1000'))
    parser.add_argument('--rlpd', type=Path, default=Path('datasets/protocol/rlpd_seed1000'))
    parser.add_argument('--output-dir', type=Path, default=Path('datasets/protocol/plots'))
    args = parser.parse_args()
    plot_completion(args.dagger, args.rlpd, args.output_dir)
