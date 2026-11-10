"""Run: PYTHONPATH=src python src/unit_tests/test_observation_normalizer.py."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer


def test_normalizer():
    encoder = ObservationEncoder(num_paths=2, num_items=2)
    a = {k: np.zeros(n) for k, n in encoder.NUMERIC_FIELDS.items()}
    a.update(powerup_type=0, attachment=9, phase=3, karts_position=(),
             paths_start=[[2, 2, 2]], paths_end=[[4, 4, 4]],
             paths_distance=[[0, 10]], paths_width=[[5]],
             items_type=[0], items_position=[[10, 0, 2]])
    b = deepcopy(a)
    b['velocity'] = [2, 2, 2]
    b['items_type'] = []
    b['items_position'] = []
    normalizer = ObservationNormalizer(encoder).fit(iter([a, b]))
    va, vb = encoder.encode(a), encoder.encode(b)
    np.testing.assert_array_equal(ObservationNormalizer.identity(encoder).transform(va),va)
    na, nb = normalizer.transform(va), normalizer.transform(vb)
    np.testing.assert_allclose(na[:3], -1)
    np.testing.assert_allclose(nb[:3], 1)
    assert normalizer.mean[-20] == 10  # Empty item slot excluded from mean.
    np.testing.assert_array_equal(nb[-20:], 0)
    np.testing.assert_array_equal(na[~encoder.normalization_mask(va)],
                                  va[~encoder.normalization_mask(va)])
    assert np.isfinite(na).all()
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'normalizer.npz'
        normalizer.save(path)
        loaded = ObservationNormalizer.load(path, encoder)
        np.testing.assert_array_equal(loaded.transform(va), na)
        try:
            ObservationNormalizer.load(path, ObservationEncoder(3, 2))
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched encoder accepted')


if __name__ == '__main__':
    test_normalizer()
    print('Normalizer checks passed')
