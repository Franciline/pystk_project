"""Run: PYTHONPATH=src python src/unit_tests/test_dagger_noise.py."""
import gzip
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from main import DAgger
from test_dagger_collection import Env, Policy


def test_noise():
    with TemporaryDirectory() as directory, patch.object(DAgger, 'Agent4', Policy):
        episodes = []
        for index, probability in enumerate([1., 1., 0.]):
            path = Path(directory) / f'{index}.pkl.gz'
            metrics = DAgger.collect_episode(Env(), Policy(), beta=1, seed=4,
                max_steps=5, output=path, iteration=1, step_size=.1,
                exploration_mode='gaussian', acceleration_std=.05,
                steer_std=.1, gaussian_probability=probability)
            with gzip.open(path, 'rb') as f: ep = pickle.load(f)
            episodes.append(ep)
            assert metrics['random_steps'] == 0
            assert metrics['noise_steps'] == (5 if probability else 0)
            assert all(a['steer']==0 for a in ep['expert_actions'])
            for a in ep['actions']:
                assert 0 <= a['acceleration'] <= 1 and -1 <= a['steer'] <= 1
                assert not a['rescue']
                assert all(a[k] == ep['expert_actions'][0][k] for k in DAgger.BUTTONS)
        assert episodes[0]['actions'] == episodes[1]['actions']
        assert episodes[0]['actions'] != episodes[0]['expert_actions']
        assert episodes[2]['actions'] == episodes[2]['expert_actions']


if __name__ == '__main__':
    test_noise()
    print('Gaussian noise checks passed')
