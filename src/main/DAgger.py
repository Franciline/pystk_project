"""DAgger collection/aggregation

Policy interface: choose_action(raw_observation) -> action dictionary.
The learner adapter owns encoding and inference. Expert always sees raw inputs.
"""
import argparse
from copy import deepcopy
import csv
import json
import gzip
import pickle
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
from utils.episode_deduplication import drop_duplicates
from utils.dagger_evaluation import evaluate_policy, append_metrics, is_improvement
from utils.multitrack import (train_tracks,next_track,evaluate_tracks,track_step_limit,
    prepare_config,save_expert_reference,MapCollection,report_collection)

CONFIG_PATH = Path(__file__).with_name('dagger_config.yaml')
BUTTONS = ('brake', 'drift', 'fire', 'nitro', 'rescue')


def to_action(action):
    def scalar(key):
        value = np.asarray(action[key]).item()
        if not np.isfinite(value):
            raise ValueError(f'Non-finite action: {key}')
        return value
    result = {
        'acceleration': float(np.clip(scalar('acceleration'), 0, 1)),
        'steer': float(np.clip(scalar('steer'), -1, 1)),
        **{key: bool(scalar(key) > 0) for key in BUTTONS},
    }
    result['rescue'] = False
    return result


def collect_episode(env, learner, *, beta, seed, max_steps, output, iteration, step_size, epsilon=0.0,
                    exploration_mode='random', acceleration_std=0.05, steer_std=0.1,
                    gaussian_probability=1.0, diagnostics=None):
    """Query expert at every visited state; select which policy controls each step."""
    if not 0 <= beta <= 1 or max_steps < 1:
        raise ValueError('Require beta in [0, 1] and max_steps >= 1')
    if beta < 1 and learner is None:
        raise ValueError('A learner policy is required when beta < 1')
    if exploration_mode not in ('random', 'gaussian'):
        raise ValueError('exploration_mode must be random or gaussian')
    if (not 0 <= epsilon <= 1 or not 0 <= gaussian_probability <= 1
            or not np.isfinite([acceleration_std, steer_std]).all()
            or min(acceleration_std, steer_std) < 0):
        raise ValueError('Invalid exploration probability or standard deviation')
    random.seed(seed)
    np.random.seed(seed)

    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    expert = Agent4(env, path_lookahead=3)

    if learner is not None and hasattr(learner, 'reset_episode'):
        learner.reset_episode()

    rng = np.random.default_rng(seed)
    recorder = DatasetRecorder(env, diagnostics)

    recorder.start_episode(obs, metadata={
        'agent': 'team4', 'track': env.current_track, 'seed': seed,
        'iteration': iteration, 'beta': beta, 'epsilon': epsilon, 'max_steps': max_steps,
        'step_size': step_size, 'laps': 1, 'num_kart': 1,
        'exploration_mode': exploration_mode, 'acceleration_std': acceleration_std,
        'steer_std': steer_std, 'gaussian_probability': gaussian_probability,
    })

    expert_steps = 0
    random_steps = 0
    noise_steps = 0

    for step in range(max_steps):
        # Isolate mutable helper policies from each other and recorded inputs.
        label = to_action(expert.choose_action(deepcopy(obs)))

        use_expert = rng.random() < beta
        action = label if use_expert else to_action(learner.choose_action(deepcopy(obs)))
        explore = rng.random() < (epsilon if exploration_mode == 'random' else gaussian_probability)
        # Copy before perturbing: expert label must remain the unperturbed recommendation.
        action = deepcopy(action)
        if explore:
            if exploration_mode == 'random':
                action = to_action(env.action_space.sample())
                random_steps += 1
            else:
                action['acceleration'] = float(np.clip(
                    action['acceleration'] + rng.normal(0, acceleration_std), 0, 1))
                action['steer'] = float(np.clip(
                    action['steer'] + rng.normal(0, steer_std), -1, 1))
                noise_steps += 1
        action['rescue'] = False
        expert_steps += int(use_expert and not explore)

        next_obs, reward, terminated, truncated, _ = env.step(deepcopy(action))
        truncated = bool(truncated or (step + 1 == max_steps and not terminated))

        recorder.record_step(action, reward, next_obs, terminated, truncated, label)

        obs = next_obs

        if terminated or truncated:
            break

    metrics = {
        'iteration': iteration, 'seed': seed, 'beta': beta, 'epsilon': epsilon,
        'n_steps': step + 1, 'completed_laps': int(terminated),
        'truncated': bool(truncated), 'expert_queries': step + 1,
        'expert_steps': expert_steps, 'random_steps': random_steps, 'noise_steps': noise_steps,
        'exploration_mode': exploration_mode,
        'lap_time': (step + 1) * step_size if terminated else None,
    }
    if recorder.diagnostics is not None:
        metrics.update(recorder.diagnostics.summary())
    recorder.episode['metadata'].update(metrics)
    recorder.save_episode(output)
    return metrics


def load_initial_episodes(env, source, destination, track):
    """Copy trusted offline episodes; use saved expert labels when available.

    Missing labels are reconstructed sequentially with a fresh Team4 per episode.
    The original offline files are never modified.
    """
    episodes = []
    for original in sorted(Path(source).rglob('*.pkl.gz')):
        with gzip.open(original, 'rb') as file:
            episode = pickle.load(file)
        if episode['metadata']['track'] not in ([track] if isinstance(track,str) else track):
            continue
        if episode['metadata'].get('agent') not in ('team4', 'The Winners'):
            continue
        episode['metadata']['source_file'] = str(original.resolve())
        episodes.append(episode)
    expected = {track} if isinstance(track,str) else set(track)
    missing = expected-{e['metadata']['track'] for e in episodes}
    if missing:
        raise ValueError(f'Missing Team4 offline data for tracks: {sorted(missing)}')
    unique = drop_duplicates(episodes)
    append_metrics(destination.parent/'offline_summary.csv', [dict(track=name,
        episodes=sum(e['metadata']['track']==name for e in unique),
        transitions=sum(len(e['actions']) for e in unique if e['metadata']['track']==name))
        for name in sorted(expected)])
    print(f'Offline deduplication: {len(episodes)} -> {len(unique)} episodes; '
          f"{sum(len(e['actions']) for e in unique)} transitions retained", flush=True)
    paths = []
    for episode in unique:
        if 'expert_actions' not in episode or any(a is None for a in episode['expert_actions']):
            expert = Agent4(env, path_lookahead=3)
            episode['expert_actions'] = [expert.choose_action(deepcopy(obs))
                                         for obs in episode['observations'][:-1]]
        episode['expert_actions'] = [to_action(a) for a in episode['expert_actions']]
        # Preserve historical executed actions: they produced the recorded transitions.
        path = destination / f'episode_{len(paths) + 1}.pkl.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'xb') as file:
            pickle.dump(episode, file, protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(path)
    return paths


def run_dagger(config, *, learner=None, train_policy=None):
    """Iteration 1 initializes data; later rounds collect a minimum step budget.

    Whole episodes finish (or time out) before checking the round's step budget.
    All rounds train on the aggregate and evaluate without exploration.
    """
    config = prepare_config(config)
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(config['output_dir'])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'config.json').open('x') as file:
        json.dump(config, file, indent=2)
    tracks = train_tracks(config)
    track = tracks[0]
    initial = config['initialization']
    evaluation = config['evaluation']
    best, paths = None, []
    initial_steps, online_steps = 0, 0
    collection=MapCollection(tracks,config["online_seed_start"])
    initial_episode_counts={track:0 for track in tracks}
    save_expert_reference(config,output_dir)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=track,
                     num_kart=1, laps=1, step_size=config['step_size'])
    try:
        for iteration in range(1, config['N'] + 1):
            round_steps = 0
            map_steps = {track:0 for track in tracks}
            map_episodes = {track:0 for track in tracks}
            if iteration == 1 and initial['mode'] == 'load_offline':
                source = Path(initial['dataset'])
                if not source.is_absolute():
                    source = root / source
                paths = load_initial_episodes(env, source, output_dir / 'iteration_1', tracks)
                for path in paths:
                    with gzip.open(path, 'rb') as file:
                        initial_steps += len(pickle.load(file)['actions'])
            else:
                if iteration == 1:
                    epsilons = ([0.0] if initial['mode'] == 'single_expert' else
                                [epsilon for epsilon in initial['epsilons']
                                 for _ in range(initial['episodes_per_epsilon'])])
                    initial_schedule = [(track,epsilon) for track in tracks for epsilon in epsilons]
                episode = 0
                while (episode < len(initial_schedule) if iteration == 1 else
                       round_steps < config['steps_per_iteration']):
                    if iteration == 1:
                        track,epsilon = initial_schedule[episode]
                    else:
                        track = next_track(tracks,map_steps)
                        epsilon = config['online_epsilon']
                    env.default_track = track
                    episode_seed = (config['online_seed_start']+initial_episode_counts[track]
                                    if iteration==1 else collection.next_seed(track))
                    episode += 1
                    path = output_dir / f'iteration_{iteration}' / f'episode_{episode}.pkl.gz'
                    print(f'Iteration {iteration}/{config["N"]}, episode {episode}, '
                          f'epsilon={epsilon}, round steps={round_steps}', flush=True)
                    exploration = config.get('exploration', {}) if iteration > 1 else {}
                    metrics = collect_episode(env, learner, beta=1.0 if iteration == 1 else 0.0,
                        seed=episode_seed, max_steps=track_step_limit(config,track), output=path,
                        iteration=iteration, step_size=config['step_size'], epsilon=epsilon,
                        exploration_mode=exploration.get('mode', 'random'),
                        acceleration_std=exploration.get('acceleration_std', 0.05),
                        steer_std=exploration.get('steer_std', 0.1),
                        gaussian_probability=exploration.get('probability', 1.0),
                        diagnostics=config.get('diagnostics'))
                    paths.append(path)
                    round_steps += metrics['n_steps']
                    map_steps[track] += metrics['n_steps']
                    if iteration == 1:
                        initial_steps += metrics['n_steps']
                        initial_episode_counts[track]+=1
                    else:
                        online_steps += metrics['n_steps']
                        collection.record(track,metrics['n_steps'])
                        map_episodes[track]+=1
                    append_metrics(output_dir / 'metrics.csv', [dict(
                        **metrics, track=track, iteration_steps=round_steps,
                        initial_steps=initial_steps, online_steps=online_steps,
                        aggregate_transitions=initial_steps + online_steps)])
            online_map_steps=map_steps if iteration>1 else {track:0 for track in tracks}
            report_collection(output_dir/'collection_by_track.csv',
                collection.summary(map_episodes,online_map_steps,iteration=iteration,online_steps=online_steps))
            with (output_dir / 'episodes.json').open('w') as file:
                json.dump([str(p.relative_to(output_dir)) for p in paths], file, indent=2)
            train_policy(learner, tuple(paths), iteration)
            checkpoint = output_dir / f'actor_iteration_{iteration}.pt'
            shutil.copyfile(checkpoint, output_dir / 'actor_last.pt')
            rows, map_summaries, summary = evaluate_tracks(env,learner,config,
                output_dir/'evaluation_episodes'/f'iteration_{iteration}',to_action)
            append_metrics(output_dir/'evaluation_by_track.csv',
                [dict(iteration=iteration,online_steps=online_steps,**s) for s in map_summaries])
            improved = is_improvement(summary, best, evaluation['min_steps_improvement'])
            append_metrics(output_dir / 'evaluation_episodes.csv',
                           [dict(iteration=iteration, **row) for row in rows])
            append_metrics(output_dir / 'evaluation.csv', [dict(
                iteration=iteration, track=tracks[0] if len(tracks)==1 else '__train_mean__', initial_steps=initial_steps,
                online_steps=online_steps, training_steps=initial_steps + online_steps,
                **summary, improved=improved)])
            print(f'Evaluation {iteration}: completion={summary["completion_rate"]:.0%}, '
                  f'finish steps={summary["finish_steps_mean"]}', flush=True)
            if improved:
                best = summary
                shutil.copyfile(checkpoint, output_dir / 'actor_best.pt')
                with (output_dir / 'best_evaluation.json').open('w') as file:
                    json.dump(dict(iteration=iteration,online_steps=online_steps,
                        selection_tracks=tracks,selection_rule='Train-map completion, then macro finishing steps; unseen excluded',
                        **summary),file,indent=2)
    finally:
        env.close()
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--experiment',type=Path,help='Shared multi-map overrides')
    args, overrides = parser.parse_known_args()
    config = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.load(args.experiment) if args.experiment else {},
        OmegaConf.from_dotlist(overrides)), resolve=True)
    from agents.dagger_actor import ActorPolicy, train_policy
    output_dir = Path(config['output_dir'])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    learner = ActorPolicy(config['actor'], output_dir, config['seed'])
    run_dagger(config, learner=learner, train_policy=train_policy)
