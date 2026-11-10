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
    config = json.loads((checkpoint.parent / 'config.json').read_text())
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
    fig, ax = plt.subplots(figsize=(max(8, len(tracks) * 4), 5))
    for model_index, (model, color) in enumerate([('DAgger', 'tab:blue'), ('RLPD', 'tab:orange')]):
        for track_index, track in enumerate(tracks):
            episode = rows[(rows['model'] == model) & (rows['track'] == track)].sort_values('seed')
            position = track_index + (-.18 if model_index == 0 else .18)
            box = ax.boxplot([episode['n_steps']], positions=[position], widths=.28,
                             patch_artist=True, showfliers=False, manage_ticks=False)
            box['boxes'][0].set(facecolor=color, alpha=.3)
            offsets = np.linspace(-.06, .06, len(episode))
            finished = episode['terminated'].to_numpy()
            for mask, marker in [(finished, 'o'), (~finished, 'x')]:
                ax.scatter(position + offsets[mask], episode['n_steps'].to_numpy()[mask],
                           color=color, marker=marker, s=35, zorder=3)
            row = summary[(summary['model'] == model) & (summary['track'] == track)].iloc[0]
            ax.annotate(f"{int(row['completed'])}/{int(row['episodes'])} finish", (position, 1),
                        xycoords=('data', 'axes fraction'), xytext=(0, 6),
                        textcoords='offset points', ha='center', fontsize=9, color=color)
        ax.plot([], [], color=color, linewidth=5, alpha=.6, label=model)
    ax.set_xticks(range(len(tracks)), tracks)
    ax.set(xlabel='Map', ylabel='Episode steps (timeouts included; lower is faster)',
           title='Selected checkpoints: generalization across maps', ylim=(0, rows['n_steps'].max() * 1.15))
    ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.text(.5, .01, 'Dots: completed laps; crosses: incomplete laps. Boxes include timeouts; completion counts shown above.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for extension in ('svg',):
        path = output / f'generalization_boxplots.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
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
