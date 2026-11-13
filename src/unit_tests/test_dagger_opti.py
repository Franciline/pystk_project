"""Checks for optional DAgger optimizations. Add --live for native simulator equivalence."""
import argparse
from copy import deepcopy
import gzip
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.dagger_actor import ActorPolicy as Original
from agents.dagger_actor_opti import ActorPolicy as Cached
from main.DAgger import to_action
from utils.multitrack import evaluate_tracks as sequential
from utils import multitrack_opti as parallel

ROOT = Path(__file__).resolve().parents[2]


def small_episodes(directory):
    paths = []
    for track in ('abyss', 'volcano_island'):
        source = sorted((ROOT/'datasets/multimap/offline'/track).rglob('*.pkl.gz'))[0]
        with gzip.open(source, 'rb') as file:
            episode = pickle.load(file)
        episode['observations'] = episode['observations'][:9]
        for key in ('actions', 'expert_actions', 'rewards', 'terminated', 'truncated'):
            episode[key] = episode[key][:8]
        path = directory / f'{track}.pkl.gz'
        with gzip.open(path, 'wb') as file:
            pickle.dump(episode, file)
        paths.append(path)
    return paths


def check_training(directory, paths):
    config = dict(num_paths=2, num_items=2, hidden_sizes=[8], device='cpu', epochs=2,
                  batch_size=4, learning_rate=.001, continuous_weight=1., boolean_weight=1.)
    torch.set_num_threads(1)
    original = Original(config, directory/'original', 1000)
    cached = Cached(config, directory/'cached', 1000)
    for iteration in (1, 2):
        original.train_iteration(paths[:iteration], iteration)
        cached.train_iteration(paths[:iteration], iteration)
        for a, b in zip(original.model.parameters(), cached.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        np.testing.assert_array_equal(original.normalizer.mean, cached.normalizer.mean)
        np.testing.assert_array_equal(original.normalizer.scale, cached.normalizer.scale)
        assert len(cached._episode_cache) == iteration
    frozen = cached.normalizer.mean.copy()
    def no_rereads(paths):
        raise AssertionError('Cached episodes were read again')
    cached._pairs = no_rereads
    cached.train_iteration(paths, 3)
    np.testing.assert_array_equal(frozen, cached.normalizer.mean)
    loaded = Original.load(directory/'cached/actor_iteration_3.pt', device='cpu')
    with gzip.open(paths[0], 'rb') as file:
        obs = pickle.load(file)['observations'][0]
    assert loaded.choose_action(obs) == cached.choose_action(obs)
    print('PASS: identical CPU weights across growing datasets, frozen statistics, no rereads, compatible checkpoint')
    return cached


def check_live(directory, policy):
    from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
    config = dict(train_tracks=['abyss'], unseen_tracks=['volcano_island'], step_size=.1,
        max_steps=60, evaluation=dict(seeds=[10000,10001]),
        diagnostics=dict(enabled=True, stuck_seconds=5, stuck_radius_m=1, save_evaluation_episodes=True),
        optimization=dict(evaluation_workers=2))
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track='abyss', num_kart=1,
                     laps=1, step_size=.1)
    try:
        expected = sequential(env, policy, config, directory/'sequential', to_action)
        parallel.start_workers(config)
        actual = parallel.evaluate_tracks(env, policy, config, directory/'parallel', to_action)
        assert expected == actual, (expected, actual)
        # Pool remains usable for the next iteration; results remain in stable map/seed order.
        again = parallel.evaluate_tracks(env, policy, config, directory/'parallel_again', to_action)
        assert again == expected
        for first in sorted((directory/'sequential').rglob('*.pkl.gz')):
            second = directory/'parallel'/first.relative_to(directory/'sequential')
            with gzip.open(first,'rb') as f: a=pickle.load(f)
            with gzip.open(second,'rb') as f: b=pickle.load(f)
            assert a['diagnostic_summary'] == b['diagnostic_summary']
            for x, y in zip(a['actions'], b['actions']):
                for key in x:
                    np.testing.assert_allclose(x[key], y[key], rtol=1e-6, atol=1e-6, err_msg=f'action {key}')
            for x,y in zip(a['observations'], b['observations']):
                for key in x:
                    if isinstance(x[key], str):
                        assert x[key] == y[key]
                    else:
                        np.testing.assert_allclose(x[key], y[key], rtol=1e-6, atol=1e-6, err_msg=f'observation {key}')
        print('PASS: sequential/2-worker metrics, actions, observations and diagnostics match; pool reused')
    finally:
        parallel.stop_workers()
        env.close()
    # Exercise the actual optional runner, aggregation, timings and checkpoint selection.
    from main.DAgger_opti import run_dagger
    from agents.dagger_actor_opti import train_policy
    output = directory/'run'
    c = dict(config, train_tracks=['abyss','volcano_island'], unseen_tracks=[], N=2,
        seed=1000, online_seed_start=1000, online_epsilon=.1, steps_per_iteration=4, max_steps=3,
        output_dir=str(output), initialization=dict(mode='load_offline',dataset=str(directory/'episodes')),
        evaluation=dict(seeds=[10000,10001],min_steps_improvement=0),
        exploration=dict(mode='gaussian',probability=1.,acceleration_std=.1,steer_std=.05))
    learner = Cached(dict(policy.config,epochs=1),output,1000)
    run_dagger(c, learner=learner, train_policy=train_policy)
    import csv
    for filename,count in [('timings.csv',2),('evaluation_by_track.csv',4),('collection_by_track.csv',4)]:
        with (output/filename).open() as file:
            assert len(list(csv.DictReader(file))) == count
    assert (output/'actor_best.pt').exists() and (output/'actor_last.pt').exists()
    print('PASS: two-iteration optimized runner, timings, aggregation and best/last checkpoints')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    with TemporaryDirectory(prefix='dagger-opti-') as temp:
        directory=Path(temp)
        (directory/'episodes').mkdir()
        paths=small_episodes(directory/'episodes')
        policy=check_training(directory,paths)
        if args.live:
            check_live(directory,policy)
