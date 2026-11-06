"""Run with PYTHONPATH=src python src/unit_tests/test_dagger_collection.py."""
import gzip
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from main import DAgger


class Policy:
    def __init__(self, env=None, path_lookahead=3):
        self.steer = 0 if env is not None else 1

    def choose_action(self, obs):
        return dict(acceleration=1, steer=self.steer,
                    brake=False, drift=False, fire=False, nitro=False, rescue=False)


class Env:
    current_track = 'test'
    action_space = type('Space', (), {'seed': lambda self, seed: None})()

    def reset(self, seed):
        self.obs = {'x': [0]}
        return self.obs, {}

    def step(self, action):
        self.obs['x'][0] += 1  # Deliberately reuse mutable observations.
        return self.obs, 1, False, False, {}


def test_collection():
    with TemporaryDirectory() as directory, patch.object(DAgger, 'Agent4', Policy):
        for beta in (0, 1):
            path = Path(directory) / f'{beta}.pkl.gz'
            metrics = DAgger.collect_episode(Env(), Policy(), beta=beta, seed=1,
                max_steps=2, output=path, iteration=1, step_size=0.1)
            with gzip.open(path, 'rb') as f:
                ep = pickle.load(f)
            assert [o['x'][0] for o in ep['observations']] == [0, 1, 2]
            assert ep['actions'][0]['steer'] == (0 if beta else 1)
            assert ep['expert_actions'][0]['steer'] == 0
            assert ep['truncated'] == [False, True]
            assert metrics['expert_queries'] == 2
            assert metrics['expert_steps'] == 2 * beta
    try:
        DAgger.run_dagger(dict(N=2, T=1, max_steps=2, initial_trajectories=1,
                               seed=0, train_tracks=['abyss'], unseen_tracks=[], step_size=0.1, output_dir='unused'))
    except ValueError:
        pass
    else:
        raise AssertionError('Missing learner accepted')


if __name__ == '__main__':
    test_collection()
    print('DAgger collection checks passed')
