"""Evaluate DAgger/RLPD checkpoints across maps and plot grouped episode-step boxes."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.dagger_actor import ActorPolicy
from agents.rlpd_agent import RLPDAgent
from main.DAgger import to_action
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.dagger_evaluation import evaluate_policy, append_metrics
from utils.multitrack import track_step_limit
from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer

ROOT = Path(__file__).resolve().parents[2]


def load_rlpd(checkpoint, device):
    # RLPD checkpoints contain normalization; no separate normalizer file needed.
    config_path = checkpoint.parent / 'config.json'
    if not config_path.exists():
        selection = json.loads((checkpoint.parent / 'best_actors_selection.json').read_text())
        config_path = Path(selection['rlpd']['source']).parent / 'config.json'
    config = json.loads(config_path.read_text())
    config['rlpd']['device'] = device
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    normalizer = ObservationNormalizer(encoder)
    normalizer.mean = state['normalizer_mean'].cpu().numpy().copy()
    normalizer.scale = state['normalizer_scale'].cpu().numpy().copy()
    policy = RLPDAgent(config, encoder, normalizer, config['seed'])
    policy.actor.load_state_dict(state['actor'])
    policy.actor.eval()
    return policy


def plot_results(output):
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
    fig, axes = plt.subplots(1, len(tracks), figsize=(19, 4.5), squeeze=False)
    labels = {'abyss': 'Abyss', 'volcano_island': 'Volcan Island', 'sandtrack': 'Shifting Sands',
              'snowtuxpeak': 'Snow Peak', 'xr591': 'XR591', 'zengarden': 'Zen Garden'}
    colors = {'DAgger': 'tab:blue', 'RLPD': 'tab:orange'}
    for ax, track in zip(axes[0], tracks):
        for position, (model, color) in enumerate(colors.items()):
            episode = rows[(rows['model'] == model) & (rows['track'] == track)].sort_values('seed')
            box = ax.boxplot([episode['n_steps']], positions=[position], widths=.35,
                             patch_artist=True, showfliers=False, manage_ticks=False,
                             medianprops=dict(color=color, linewidth=1.8))
            box['boxes'][0].set(facecolor=color, alpha=.3)
            offsets = np.linspace(-.07, .07, len(episode))
            finished = episode['terminated'].to_numpy()
            for mask, marker in [(finished, 'o'), (~finished, 'x')]:
                ax.scatter(position + offsets[mask], episode['n_steps'].to_numpy()[mask],
                           color=color, marker=marker, s=20, zorder=3, alpha=.75)
            ax.text(position, 1.02, f'{finished.sum()}/{len(episode)} finish',
                    transform=ax.get_xaxis_transform(), ha='center', fontsize=9, color=color)
        ax.axhline(expert.loc[track, 'finish_steps_mean'], color='#333333', linestyle='--', linewidth=1.3)
        ax.set_xticks([0, 1], ['DAgger', 'RLPD'])
        ax.set_xlim(-.5, 1.5)
        ax.set_title(labels.get(track, track), pad=30)
        ax.set_ylabel('Episode steps')
        ax.margins(y=.15)
    handles = [Patch(facecolor=c, alpha=.5, label=m) for m, c in colors.items()]
    handles += [Line2D([], [], color='#333333', linestyle='--', label='Expert (team4) mean'),
                Line2D([], [], color='gray', marker='o', linestyle='', label='Completed lap'),
                Line2D([], [], color='gray', marker='x', linestyle='', label='Incomplete lap (timeout)')]
    fig.suptitle('DAgger and RLPD across maps - 20 evaluation seeds', y=.99)
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, 0))
    fig.tight_layout(rect=(0, .09, 1, .94))
    path = output / 'generalization_boxplots.svg'
    fig.savefig(path, bbox_inches='tight')
    print(path)
    plt.close(fig)


def compare(config, dagger, rlpd, device, output):
    settings = config['comparison']
    # Load first, so a missing checkpoint does not leave a partial experiment folder.
    policies = {'DAgger': ActorPolicy.load(dagger, device=device), 'RLPD': load_rlpd(rlpd, device)}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'config.json').open('x') as file:
        json.dump(dict(comparison=settings, step_size=config['step_size'],
                       max_steps=config['max_steps'],max_steps_by_track=config.get('max_steps_by_track',{}),dagger=str(dagger.resolve()),
                       rlpd=str(rlpd.resolve()), device=device), file, indent=2)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False),
                     track=settings['tracks'][0], num_kart=1, laps=1,
                     step_size=config['step_size'])
    try:
        for track in settings['tracks']:
            env.default_track = track
            for name, policy in policies.items():
                print(f'{name} on {track}: {len(settings["seeds"])} episodes', flush=True)
                rows, summary = evaluate_policy(env, policy, settings['seeds'], track_step_limit(config,track), to_action)
                append_metrics(output / 'episodes.csv', [dict(model=name, track=track, **r) for r in rows])
                append_metrics(output / 'summary.csv', [dict(model=name, track=track, **summary)])
                print(f"Completed {summary['completed']}/{summary['episodes']}; mean steps={summary['steps_mean']:.1f}", flush=True)
    finally:
        env.close()
    from main.benchmark_expert import run_benchmark
    expert_config = dict(config, train_tracks=settings['tracks'], unseen_tracks=[])
    run_benchmark(expert_config, output / 'expert', settings['seeds'])
    plot_results(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dagger', type=Path, help='DAgger actor_best.pt path')
    parser.add_argument('--rlpd', type=Path, help='RLPD agent_best.pt path; keep training config.json beside it')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('dagger_config.yaml'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['auto','cpu','mps'], default='auto')
    parser.add_argument('--plot-only', action='store_true', help='Replot saved CSVs without simulator rollouts')
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    output = args.output_dir or Path(config['comparison']['output_dir'])
    if not output.is_absolute(): output = ROOT / output
    if args.plot_only:
        plot_results(output)
    else:
        if args.dagger is None or args.rlpd is None:
            parser.error('--dagger and --rlpd checkpoints are required unless --plot-only')
        compare(config, args.dagger, args.rlpd, args.device, output)
