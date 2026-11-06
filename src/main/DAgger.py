"""DAgger collection/aggregation

Policy interface: choose_action(raw_observation) -> action dictionary.
The learner adapter owns encoding and inference. Expert always sees raw inputs.
"""
import argparse
from copy import deepcopy
import csv
import json
import shutil
from pathlib import Path
import random
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.team4.agent4 import Agent4
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.dataset_recorder import DatasetRecorder
from utils.dagger_evaluation import evaluate_policy, append_metrics

CONFIG_PATH = Path(__file__).with_name('dagger_config.yaml')
BUTTONS = ('brake', 'drift', 'fire', 'nitro', 'rescue')


def to_action(action):
    def scalar(key):
        value = np.asarray(action[key]).item()
        if not np.isfinite(value):
            raise ValueError(f'Non-finite action: {key}')
        return value
    return {
        'acceleration': float(np.clip(scalar('acceleration'), 0, 1)),
        'steer': float(np.clip(scalar('steer'), -1, 1)),
        **{key: bool(scalar(key) > 0) for key in BUTTONS},
    }


def collect_episode(env, learner, *, beta, seed, max_steps, output, iteration, step_size):
    """Query expert at every visited state; select which policy controls each step."""
    if not 0 <= beta <= 1 or max_steps < 1:
        raise ValueError('Require beta in [0, 1] and max_steps >= 1')
    if beta < 1 and learner is None:
        raise ValueError('A learner policy is required when beta < 1')
    random.seed(seed)
    np.random.seed(seed)

    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    expert = Agent4(env, path_lookahead=3)

    if learner is not None and hasattr(learner, 'reset_episode'):
        learner.reset_episode()

    rng = np.random.default_rng(seed)
    recorder = DatasetRecorder()

    recorder.start_episode(obs, metadata={
        'agent': 'team4', 'track': env.current_track, 'seed': seed,
        'iteration': iteration, 'beta': beta, 'max_steps': max_steps,
        'step_size': step_size, 'laps': 1, 'num_kart': 1,
    })

    expert_steps = 0

    for step in range(max_steps):
        # Isolate mutable helper policies from each other and recorded inputs.
        label = to_action(expert.choose_action(deepcopy(obs)))

        use_expert = rng.random() < beta
        action = label if use_expert else to_action(learner.choose_action(deepcopy(obs)))
        expert_steps += int(use_expert)

        next_obs, reward, terminated, truncated, _ = env.step(deepcopy(action))
        truncated = bool(truncated or (step + 1 == max_steps and not terminated))

        recorder.record_step(action, reward, next_obs, terminated, truncated, label)

        obs = next_obs

        if terminated or truncated:
            break

    metrics = {
        'iteration': iteration, 'seed': seed, 'beta': beta,
        'n_steps': step + 1, 'completed_laps': int(terminated),
        'truncated': bool(truncated), 'expert_queries': step + 1,
        'expert_steps': expert_steps,
        'lap_time': (step + 1) * step_size if terminated else None,
    }
    recorder.episode['metadata'].update(metrics)
    recorder.save_episode(output)
    return metrics


def run_dagger(config, *, learner=None, train_policy=None):
    """Aggregate episode paths; train_policy(learner, paths, iteration) updates actor.

    Iteration 1 is expert-only; later iterations are learner-only, with expert labels.
    Multi-iteration runs require training code; no untrained/random substitute.
    """
    iterations, trajectories = config['N'], config['T']
    max_steps = config['max_steps']
    initial_trajectories = config['initial_trajectories']
    seed, step_size = config['seed'], config['step_size']
    tracks, unseen = config['train_tracks'], config['unseen_tracks']
    if not tracks or len(set(tracks + unseen)) != len(tracks + unseen):
        raise ValueError('Require nonempty training tracks and unique disjoint track lists')
    track = tracks[0]
    output_dir = Path(config['output_dir'])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    if step_size <= 0:
        raise ValueError('step_size must be positive')
    if iterations < 1 or trajectories < 1 or max_steps < 1 or initial_trajectories < 1:
        raise ValueError('Invalid iteration, trajectory, step, or beta settings')
    
    if iterations > 1 and (learner is None or train_policy is None):
        raise ValueError('Multiple iterations require learner and train_policy')
    
    evaluation = config.get('evaluation')
    if evaluation is not None:
        if learner is None or train_policy is None:
            raise ValueError('Evaluation requires a learner and training callback')
        seeds = evaluation['seeds']
        if (not seeds or len(set(seeds)) != len(seeds)
                or evaluation['min_relative_improvement'] < 0):
            raise ValueError('Invalid evaluation settings')
        if any(seed <= s < seed + len(tracks) * (initial_trajectories + (iterations - 1) * trajectories) for s in seeds):
            raise ValueError('Evaluation seeds overlap collection seeds')
    best = None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Exclusive creation prevents mixing separate experiments accidentally.
    with (output_dir / 'config.json').open('x') as f:
        json.dump(config, f, indent=2)

    baselines = None
    if evaluation is not None:
        from main.benchmark_expert import run_benchmark
        start = config['expert_benchmark']['seed_start']
        runs = config['expert_benchmark']['runs']
        if runs < 1:
            raise ValueError('Expert benchmark runs must be positive')
        baselines = run_benchmark(config, output_dir / 'expert_benchmark', list(range(start, start + runs)))
        missing = [t for t in tracks if not baselines[t]['finish_steps_mean']]
        if missing:
            raise ValueError(f'Expert never finished training tracks: {missing}; relative scoring unavailable')
    paths = []
    total = 0
    collected = 0
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False),
                     track=track, num_kart=1, laps=1, step_size=step_size)
    try:
        with (output_dir / 'metrics.csv').open('x', newline='') as f:
            writer = None
            for iteration in range(1, iterations + 1):
                beta = 1.0 if iteration == 1 else 0.0
                episode_count = initial_trajectories if iteration == 1 else trajectories
                for track in tracks:
                    env.default_track = track
                    for episode in range(episode_count):
                        episode_seed = seed + collected
                        collected += 1
                        path = output_dir / f'iteration_{iteration}' / track / f'episode_{episode + 1}.pkl.gz'
                        print(f'Iteration {iteration}/{iterations}, track={track}, episode {episode + 1}/{episode_count}, beta={beta:.3f}', flush=True)
                        metrics = collect_episode(env, learner, beta=beta, seed=episode_seed,
                                                  max_steps=max_steps, output=path, iteration=iteration,
                                                  step_size=step_size)
                        metrics['track'] = track
                        paths.append(path)
                        metrics['aggregate_transitions'] = total + metrics['n_steps']
                        total = metrics['aggregate_transitions']

                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(metrics))
                            writer.writeheader()
                        writer.writerow(metrics)
                        f.flush()
                    
                        with (output_dir / 'episodes.json').open('w') as manifest:
                            json.dump([str(p.relative_to(output_dir)) for p in paths], manifest, indent=2)

                if train_policy is not None:
                    train_policy(learner, tuple(paths), iteration)

                if evaluation is not None:
                    summaries, finished_keys, ratios = {}, [], []
                    try:
                        for eval_track in tracks + unseen:
                            env.default_track = eval_track
                            rows, summary = evaluate_policy(env, learner, seeds, max_steps, to_action)
                            baseline = baselines[eval_track]['finish_steps_mean']
                            summary['relative_finish_steps'] = (
                                summary['finish_steps_mean'] / baseline
                                if summary['finish_steps_mean'] is not None and baseline else None)
                            summaries[eval_track] = summary
                            if eval_track in tracks:
                                for row in rows:
                                    if row['terminated']:
                                        finished_keys.append((eval_track, row['seed']))
                                ratios.append(summary['relative_finish_steps'])
                            append_metrics(output_dir / 'evaluation_episodes.csv',
                                [dict(iteration=iteration, track=eval_track, **row) for row in rows])
                            append_metrics(output_dir / 'evaluation.csv', [dict(
                                iteration=iteration, track=eval_track, training_steps=total,
                                **summary, selection_track=eval_track in tracks)])
                            print(f"Evaluation {iteration}, {eval_track}: completion={summary['completion_rate']:.0%}, "
                                  f"relative steps={summary['relative_finish_steps']}", flush=True)
                        score = dict(completion_rate=float(np.mean([summaries[t]['completion_rate'] for t in tracks])),
                                     relative_finish_steps=float(np.mean(ratios)) if all(r is not None for r in ratios) else None,
                                     completed_episodes=finished_keys)
                        improved = best is None or score['completion_rate'] > best['completion_rate']
                        if best is not None and score['completion_rate'] == best['completion_rate']:
                            if (score['completed_episodes'] == best['completed_episodes']
                                    and score['relative_finish_steps'] is not None
                                    and best['relative_finish_steps'] is not None):
                                gain = best['relative_finish_steps'] - score['relative_finish_steps']
                                improved = gain > 0 and gain >= evaluation['min_relative_improvement']
                        append_metrics(output_dir / 'evaluation_aggregate.csv', [dict(
                            iteration=iteration, training_steps=total,
                            completion_rate=score['completion_rate'],
                            relative_finish_steps=score['relative_finish_steps'], improved=improved)])
                        if improved:
                            best = score
                            shutil.copyfile(output_dir / f'actor_iteration_{iteration}.pt', output_dir / 'actor_best.pt')
                            with (output_dir / 'best_evaluation.json').open('w') as file:
                                json.dump(dict(iteration=iteration, **score), file, indent=2)
                    finally:
                        env.default_track = tracks[0]
    finally:
        env.close()
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    from agents.dagger_actor import ActorPolicy, train_policy
    output_dir = Path(config['output_dir'])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    learner = ActorPolicy(config['actor'], output_dir, config['seed'])
    run_dagger(config, learner=learner, train_policy=train_policy)
