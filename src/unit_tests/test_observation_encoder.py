"""Run with PYTHONPATH=src python src/unit_tests/test_observation_encoder.py."""
from copy import deepcopy
import numpy as np
from utils.observation_encoder import ObservationEncoder


def test_encoding():
    encoder = ObservationEncoder(num_paths=2, num_items=2)
    obs = {k: np.zeros(n) for k, n in encoder.NUMERIC_FIELDS.items()}
    obs.update(powerup_type=0, attachment=9, phase=3, karts_position=(),
               paths_start=[[1, 2, 3]], paths_end=[[4, 5, 6]],
               paths_distance=[[0, 10]], paths_width=[[5]],
               items_type=[0], items_position=[[1, 0, 2]], kart_type='tux')
    original = deepcopy(obs)
    vector = encoder.encode(obs)
    assert vector.shape == (encoder.output_size,)
    rows = vector[-20:].reshape(2, 10)
    assert rows[0, 3] == 1 and rows[0, -1] == 1  # Real bonus box.
    assert np.all(rows[1] == 0)  # Padding is not a bonus box.
    assert np.array_equal(obs['velocity'], original['velocity'])
    assert obs['items_type'] == original['items_type']
    obs['items_type'] = [0, 4, 1]
    obs['items_position'] = [[1, 0, 2]] * 3
    assert encoder.encode(obs).shape == vector.shape
    obs['items_type'][0] = 99
    try:
        encoder.encode(obs)
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown category accepted')


if __name__ == '__main__':
    test_encoding()
    print('Encoder checks passed')
