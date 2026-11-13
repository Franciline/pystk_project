"""Run with: python src/main/test_paired_test.py."""
import numpy as np

from paired_test import sign_flip_p


if __name__ == '__main__':
    # Only the all-positive/all-negative patterns match the observed magnitude.
    assert sign_flip_p(np.ones(4)) == 2 / 2 ** 4
    assert sign_flip_p(np.array([1., -1.])) == 1
    assert sign_flip_p(np.zeros(4)) == 1
    print('Paired sign-flip checks passed')
