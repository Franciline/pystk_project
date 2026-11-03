"""Run: PYTHONPATH=src python src/unit_tests/test_dagger_actor.py."""
import csv
import gzip
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from agents.dagger_actor import ActorPolicy


def test_actor():
    config = dict(num_paths=2, num_items=2, hidden_sizes=[16], learning_rate=0.02,
                  batch_size=8, epochs=15, continuous_weight=1., boolean_weight=1., device='cpu')
    with TemporaryDirectory() as directory:
        root = Path(directory)
        policy = ActorPolicy(config, root, seed=1)
        obs = {k: np.zeros(n) for k, n in policy.encoder.NUMERIC_FIELDS.items()}
        obs.update(powerup_type=0, attachment=9, phase=3, karts_position=(),
                   paths_start=[[1, 0, 2]], paths_end=[[1, 0, 3]],
                   paths_distance=[[0, 1]], paths_width=[[5]],
                   items_type=[], items_position=[])
        label = dict(acceleration=1., steer=-.5, brake=False, drift=True,
                     fire=False, nitro=True, rescue=False)
        path = root / 'episode.pkl.gz'
        with gzip.open(path, 'wb') as f:
            pickle.dump(dict(observations=[obs] * 17, expert_actions=[label] * 16), f)
        policy.train_iteration([path], 1)
        with (root / 'losses.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert float(rows[-1]['loss']) < float(rows[0]['loss'])
        action = policy.choose_action(obs)
        assert 0 <= action['acceleration'] <= 1 and -1 <= action['steer'] <= 1
        assert all(isinstance(action[k], bool) for k in ('brake','drift','fire','nitro','rescue'))
        mean = policy.normalizer.mean.copy()
        policy.config['epochs'] = 1
        policy.train_iteration([path], 2)
        np.testing.assert_array_equal(policy.normalizer.mean, mean)
        checkpoint = torch.load(root / 'actor_iteration_2.pt', weights_only=True)
        other = ActorPolicy(config, root / 'other', seed=2)
        other.model.load_state_dict(checkpoint['model'])
        other.normalizer.mean = checkpoint['normalizer_mean'].numpy()
        other.normalizer.scale = checkpoint['normalizer_scale'].numpy()
        assert other.choose_action(obs) == policy.choose_action(obs)


if __name__ == '__main__':
    test_actor()
    print('Actor training checks passed')
