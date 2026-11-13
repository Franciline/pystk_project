"""Evaluate saved policies without implicitly plotting results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
import hashlib
from agents.dagger_actor import ActorPolicy
from agents.rlpd_agent import RLPDAgent
from agents.team4.agent4 import Agent4
from main.DAgger import to_action
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.dagger_evaluation import evaluate_policy, append_metrics
from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer
from utils.multitrack import track_step_limit
single_PROTOCOL = {'track': 'abyss', 'step_size': 0.1, 'max_steps': 1000}
single_EXPERT_DIR = Path(__file__).resolve().parents[2] / 'datasets' / 'protocol' / 'expert_test'

def single_load_policy(run_dir, name, device='auto'):
    config = json.loads((run_dir / 'config.json').read_text())
    if any((config.get(key) != value for key, value in single_PROTOCOL.items())):
        raise ValueError(f'{run_dir} was not trained with the single-map protocol')
    if name == 'bc':
        return ActorPolicy.load(run_dir / 'actor_iteration_1.pt', device=device)
    candidates = [run_dir / f'actor_{name}.pt', run_dir / f'agent_{name}.pt']
    existing = [p for p in candidates if p.exists()]
    if len(existing) != 1:
        raise ValueError(f'Expected exactly one actor/agent {name} checkpoint at {run_dir}')
    checkpoint = existing[0]
    if checkpoint_method(checkpoint) == 'dagger':
        return ActorPolicy.load(checkpoint, device=device)
    return comparison_load_rlpd(checkpoint, device)

def single_test(env, out_dir, checkpoint, policy, seeds):
    """race one policy on every seed, save each race and the summary"""
    expert = (lambda env: Agent4(env, path_lookahead=3)) if policy is None else None
    rows, summary = evaluate_policy(env, policy, seeds, single_PROTOCOL['max_steps'], to_action, policy_factory=expert)
    append_metrics(out_dir / 'final_test_episodes.csv', [dict(checkpoint=checkpoint, **r) for r in rows])
    append_metrics(out_dir / 'final_test.csv', [dict(checkpoint=checkpoint, **summary)])
    print(f"{out_dir.name} {checkpoint}: completion={summary['completion_rate']:.0%}, finish steps={summary['finish_steps_mean']}", flush=True)
comparison_ROOT = Path(__file__).resolve().parents[2]

def comparison_load_rlpd(checkpoint, device):
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

def comparison_compare(config, dagger, rlpd, device, output):
    settings = config['comparison']
    policies = {'DAgger': ActorPolicy.load(dagger, device=device), 'RLPD': comparison_load_rlpd(rlpd, device)}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'config.json').open('x') as file:
        json.dump(dict(comparison=settings, step_size=config['step_size'], max_steps=config['max_steps'], max_steps_by_track=config.get('max_steps_by_track', {}), dagger=str(dagger.resolve()), rlpd=str(rlpd.resolve()), device=device), file, indent=2)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=settings['tracks'][0], num_kart=1, laps=1, step_size=config['step_size'])
    try:
        for track in settings['tracks']:
            env.default_track = track
            for name, policy in policies.items():
                print(f"{name} on {track}: {len(settings['seeds'])} episodes", flush=True)
                rows, summary = evaluate_policy(env, policy, settings['seeds'], track_step_limit(config, track), to_action)
                append_metrics(output / 'episodes.csv', [dict(model=name, track=track, **r) for r in rows])
                append_metrics(output / 'summary.csv', [dict(model=name, track=track, **summary)])
                print(f"Completed {summary['completed']}/{summary['episodes']}; mean steps={summary['steps_mean']:.1f}", flush=True)
    finally:
        env.close()
    from main.benchmark_expert import run_benchmark
    expert_config = dict(config, train_tracks=settings['tracks'], unseen_tracks=[])
    run_benchmark(expert_config, output / 'expert', settings['seeds'])
bc_ROOT = Path(__file__).resolve().parents[2]

def checkpoint_method(path):
    """Shared YAMLs are not evidence of the checkpoint's policy type."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'actor' in state and 'critic' in state:
        return 'rlpd'
    if 'model' in state and 'categories' in state:
        return 'dagger'
    raise ValueError(f'Unknown policy checkpoint format: {path}')

def main():
    from utils.experiment_config import CONFIG_PATH, load_config
    parser = argparse.ArgumentParser(description='Evaluate saved policies; plotting is a separate command.')
    parser.add_argument('--setup', choices=['singlemap', 'multimap', 'comparison', 'expert'], required=True)
    parser.add_argument('--run-dirs', type=Path, nargs='+')
    parser.add_argument('--checkpoints', choices=['best', 'last', 'bc'], nargs='+', default=['best'])
    parser.add_argument('--expert', action='store_true', help='Also evaluate Team4 in singlemap mode')
    parser.add_argument('--root', type=Path, default=comparison_ROOT / 'datasets/multimap/full5maps50k')
    parser.add_argument('--training-seeds', type=int, nargs='+', default=[1000, 2000, 3000])
    parser.add_argument('--seed-start', type=int, default=20000)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--profile', default='singlemap_dagger')
    parser.add_argument('--tracks', nargs='+', help='Maps for comparison/expert evaluation')
    parser.add_argument('--dagger', type=Path)
    parser.add_argument('--rlpd', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    args = parser.parse_args()
    if args.runs < 1 or args.workers < 1:
        parser.error('runs and workers must be positive')
    seeds = list(range(args.seed_start, args.seed_start + args.runs))
    if args.setup == 'multimap':
        from utils.multimap_reports import read_runs, make_jobs, evaluate_all
        if args.runs != 20 or args.seed_start != 20000:
            parser.error('The multimap report protocol uses the 20 seeds 20000..20019')
        if len(set(args.training_seeds)) != 3 or len(args.training_seeds) != 3:
            parser.error('The multimap report protocol requires three distinct training seeds')
        runs, reference = read_runs(args.root, args.training_seeds)
        evaluate_all(make_jobs(args.root, runs, reference, args.training_seeds), args.workers)
        return
    if args.setup in ('comparison', 'expert'):
        config = OmegaConf.to_container(load_config(args.config, args.profile), resolve=True)
        from utils.multitrack import prepare_config, evaluation_tracks
        config = prepare_config(config)
        if args.tracks:
            config['train_tracks'] = args.tracks
            config['unseen_tracks'] = []
        config.setdefault('comparison', {'tracks': evaluation_tracks(config), 'seeds': seeds})
        if args.tracks: config['comparison']['tracks'] = args.tracks
        if args.output_dir is None:
            parser.error('--output-dir is required for comparison/expert')
        if args.setup == 'expert':
            from main.benchmark_expert import run_benchmark
            run_benchmark(config, args.output_dir, seeds)
        else:
            if args.dagger is None or args.rlpd is None:
                parser.error('Specify --dagger and --rlpd checkpoints')
            config['comparison']['seeds'] = seeds
            comparison_compare(config, args.dagger, args.rlpd, args.device, args.output_dir)
        return
    directories = args.run_dirs or []
    if not directories and (not args.expert):
        parser.error('Specify --run-dirs or --expert')
    expert_dir = args.output_dir or single_EXPERT_DIR
    destinations = {directory: args.output_dir/directory.name if args.output_dir else directory for directory in directories}
    if args.expert and expert_dir in destinations.values():
        parser.error('Expert output must be separate from learner outputs')
    for directory in list(destinations.values()) + ([expert_dir] if args.expert else []):
        if (directory / 'final_test.csv').exists():
            raise FileExistsError(f'{directory} already tested')
    jobs = [(destinations[directory], name, single_load_policy(directory, name, args.device)) for directory in directories for name in args.checkpoints]
    if args.expert:
        expert_dir.mkdir(parents=True, exist_ok=True)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track='abyss', num_kart=1, laps=1, step_size=0.1)
    try:
        if args.expert:
            single_test(env, expert_dir, 'team4', None, seeds)
        for directory, name, policy in jobs:
            single_test(env, directory, name, policy, seeds)
    finally:
        env.close()
if __name__ == '__main__':
    main()
