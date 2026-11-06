"""Display track segments as horizontal strips measured in meters."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_segments(geometry_dir, output):
    files = sorted(Path(geometry_dir).glob('*_segments.csv'))
    if not files:
        raise ValueError(f'No segment CSVs in {geometry_dir}')
    fig, ax = plt.subplots(figsize=(16, max(3.5, len(files) * .8 + 1.8)))
    labels = []
    longest = 0
    for row, path in enumerate(files):
        name = path.name.removesuffix('_segments.csv')
        data = pd.read_csv(path).sort_values('segment_id')
        lengths = data['length_m'].to_numpy(dtype=float)
        if not len(lengths) or not np.isfinite(lengths).all() or np.any(lengths < 0):
            raise ValueError(f'Invalid segment lengths: {path}')
        boundaries = np.r_[0, np.cumsum(lengths)]
        total = boundaries[-1]
        longest = max(longest, total)
        ax.hlines(row, 0, total, color='tab:blue', linewidth=3)
        ax.vlines(boundaries, row - .19, row + .19, color='#18334b', linewidth=.55)
        metadata = path.with_name(name + '.json')
        official = json.loads(metadata.read_text())['track_length_m'] if metadata.exists() else None
        text = f'{total:,.1f} m segment sum'
        if official is not None:
            text += f' | {official:,.1f} m official'
        ax.annotate(text, (total, row), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=9)
        labels.append(f'{name} ({len(lengths)} segments)')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_xlim(0, longest * 1.4)
    ax.set_xlabel('Cumulative geometric segment length (m)')
    ax.set_title('Track segment lengths — each vertical tick marks a boundary')
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.text(.5, .015, 'Segments concatenated in segment-ID order; this is not a spatial route drawing. Branches may make the sum differ from official track length.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    # Vector version allows zooming into dense segment boundaries.
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    print(output)
    print(output.with_suffix('.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path('datasets/dagger'))
    parser.add_argument('--output', type=Path, help='Default: <experiment>/plots/track_segments.png')
    args = parser.parse_args()
    plot_segments(args.experiment / 'expert_benchmark' / 'track_geometry',
                  args.output or args.experiment / 'plots' / 'track_segments.png')
