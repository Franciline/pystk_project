"""Measure Team4 baseline performance on the configured evaluation tracks."""
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

from omegaconf import OmegaConf  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main.DAgger import CONFIG_PATH, to_action
from agents.team4.agent4 import Agent4
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.track_diagnostics import save_track_diagnostics
from utils.multitrack import evaluation_tracks, track_step_limit
from utils.dagger_evaluation import evaluate_policy, append_metrics


def run_benchmark(config, output_dir, seeds):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'config.json').open('x') as file:
        json.dump({'task_config': config, 'seeds': seeds, 'expert': 'team4'}, file, indent=2)
    tracks = evaluation_tracks(config)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False),
                     track=tracks[0], num_kart=1, laps=1, step_size=config['step_size'])
    try:
        summaries = {}
        for track in tracks:
            env.default_track = track
            rows, summary = evaluate_policy(
                env, None, seeds, track_step_limit(config,track), to_action,
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


def prepare_protocol(config):
    """Benchmark once, freeze shared limits, and reuse them across both algorithms."""
    settings=config['track_timeouts']
    root=Path(__file__).resolve().parents[2]
    directory=Path(settings['directory'])
    if not directory.is_absolute(): directory=root/directory
    benchmark=config['expert_benchmark']
    seeds=list(range(benchmark['seed_start'],benchmark['seed_start']+benchmark['runs']))
    margin=int(settings['margin_steps']); cap=int(settings['benchmark_max_steps'])
    if margin < 0 or cap < 1 or not seeds:
        raise ValueError('Require nonnegative timeout margin, positive expert cap and benchmark runs')
    signature=dict(tracks=evaluation_tracks(config),step_size=config['step_size'],
                   seeds=seeds,margin_steps=margin,benchmark_max_steps=cap)
    frozen=directory/'track_timeouts.json'
    if frozen.exists():
        manifest=json.loads(frozen.read_text())
        if manifest['protocol']!=signature:
            raise ValueError(f'Frozen expert protocol differs at {directory}; choose a new track_timeouts.directory')
        return directory,manifest['max_steps_by_track']
    if not (directory/'summary.csv').exists():
        calibration=deepcopy(config)
        calibration['max_steps']=cap
        calibration['max_steps_by_track']={}
        run_benchmark(calibration,directory,seeds)
    else:
        previous=json.loads((directory/'config.json').read_text())
        old=previous['task_config']
        if (evaluation_tracks(old)!=signature['tracks'] or old['step_size']!=signature['step_size']
                or previous['seeds']!=seeds or old['max_steps']!=cap):
            raise ValueError(f'Existing expert calibration differs at {directory}; choose a new directory')
    with (directory/'episodes.csv').open() as file:
        rows=list(csv.DictReader(file))
    limits={}; details={}
    for track in signature['tracks']:
        observed=[r for r in rows if r['track']==track]
        finished=[int(r['n_steps']) for r in observed if r['terminated'].lower()=='true']
        if len(observed)!=len(seeds) or not finished:
            raise ValueError(f'No complete expert calibration for {track} within {cap} steps; '
                             'inspect the benchmark, then choose a larger cap and a fresh reference directory')
        limits[track]=max(finished)+margin
        details[track]=dict(max_expert_finishing_steps=max(finished),completed=len(finished),episodes=len(observed))
    with frozen.open('x') as file:
        json.dump(dict(protocol=signature,max_steps_by_track=limits,expert_results=details),file,indent=2)
    print('Frozen track timeouts:',limits,flush=True)
    return directory,limits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--experiment',type=Path,help='Shared multi-map overrides')
    parser.add_argument('--freeze-timeouts',action='store_true',
                        help='Prepare/reuse the shared expert baseline and frozen per-map limits only')
    parser.add_argument('--runs',type=int,help='Override benchmark run count (default from config, otherwise 5)')
    parser.add_argument('--seed-start',type=int,help='Override benchmark seed start (default 10000)')
    parser.add_argument('--output-dir',type=Path,help='Benchmark directory; overrides shared reference when freezing')
    args=parser.parse_args()
    if (args.runs is not None and args.runs<1) or (args.seed_start is not None and args.seed_start<0):
        parser.error('Require positive --runs and nonnegative --seed-start')
    config=OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(args.config),
        OmegaConf.load(args.experiment) if args.experiment else {}),resolve=True)
    benchmark=dict(config.get('expert_benchmark') or dict(runs=5,seed_start=10000))
    if args.runs is not None:benchmark['runs']=args.runs
    if args.seed_start is not None:benchmark['seed_start']=args.seed_start
    config['expert_benchmark']=benchmark
    if args.freeze_timeouts:
        if config.get('track_timeouts',{}).get('mode')!='expert':
            parser.error('--freeze-timeouts requires track_timeouts.mode=expert in the experiment config')
        if args.output_dir is not None:config['track_timeouts']['directory']=str(args.output_dir)
        directory,limits=prepare_protocol(config)
        print(f'Shared expert reference ready: {directory}',flush=True)
        print(f'Per-map limits: {limits}',flush=True)
    else:
        output=args.output_dir or Path(__file__).resolve().parents[2]/'datasets/expert_benchmark'
        run_benchmark(config,output,list(range(benchmark['seed_start'],benchmark['seed_start']+benchmark['runs'])))
