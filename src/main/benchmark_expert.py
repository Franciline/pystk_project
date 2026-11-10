"""Measure Team4 baseline performance on the configured evaluation tracks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from main.DAgger import CONFIG_PATH, to_action
from agents.team4.agent4 import Agent4
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.track_diagnostics import save_track_diagnostics
from utils.multitrack import evaluation_tracks
from utils.dagger_evaluation import evaluate_policy, append_metrics


def run_benchmark(config, output_dir, seeds):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'config.json').open('x') as file:
        json.dump(dict(task_config=config, seeds=seeds, expert='team4'), file, indent=2)
    tracks = evaluation_tracks(config)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False),
                     track=tracks[0], num_kart=1, laps=1, step_size=config['step_size'])
    try:
        summaries = {}
        for track in tracks:
            env.default_track = track
            rows, summary = evaluate_policy(
                env, None, seeds, config['max_steps'], to_action,
                policy_factory=lambda env: Agent4(env, path_lookahead=3),
                diagnostics=config.get('diagnostics'),record_dir=output_dir/'episodes'/track)
            save_track_diagnostics(env, output_dir)
            summaries[track] = summary
            append_metrics(output_dir / 'episodes.csv', [dict(track=track, **r) for r in rows])
            append_metrics(output_dir / 'summary.csv', [dict(track=track, **summary)])
            print(f"Expert {track}: {summary['completed']}/{len(seeds)} completed, "
                  f"finishing steps={summary['finish_steps_mean']} +/- {summary['finish_steps_std']}", flush=True)
    finally:
        env.close()
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed-start', type=int, default=10000)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'datasets' / 'expert_benchmark')
    args = parser.parse_args()
    if args.runs < 1 or args.seed_start < 0:
        parser.error('Require positive --runs and nonnegative --seed-start')
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    run_benchmark(config, args.output_dir, list(range(args.seed_start, args.seed_start + args.runs)))
