"""Paired comparison on the final-test races: Delta, 95% CI and p, as in the report table.

Run from the repo root:
  python paired_test.py --method RUN_1000 RUN_2000 RUN_3000                 # vs the expert
  python paired_test.py --method RUN_A... --reference RUN_B...              # method A vs method B
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPERT = Path('datasets/protocol/expert_test')
T_975 = {16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093}  # Student t 0.975 quantile, by degrees of freedom


def race_steps(run_dirs, checkpoint):
    """steps on each race seed, one column per training run; a timeout becomes NaN"""
    runs = []
    for run in run_dirs:
        rows = pd.read_csv(Path(run) / 'final_test_episodes.csv')
        rows = rows[rows['checkpoint'] == checkpoint].set_index('seed')
        finished = rows['terminated'].astype(str).str.lower() == 'true'
        runs.append(rows['n_steps'].where(finished))
    return pd.concat(runs, axis=1)


def sign_flip_p(d):
    """exact two-sided p: share of the 2^n sign patterns whose mean is at least as far from 0"""
    n, observed, hits = len(d), abs(d.mean()), 0
    bits = np.arange(n)
    for start in range(0, 2 ** n, 2 ** 16):
        patterns = np.arange(start, min(start + 2 ** 16, 2 ** n))[:, None]
        signs = 1 - 2 * ((patterns >> bits) & 1)          # every +/- combination
        hits += np.sum(np.abs(signs @ d) / n >= observed - 1e-9)
    return hits / 2 ** n


def compare(method, reference):
    a = method.mean(axis=1, skipna=False)                 # average over training seeds, per race
    b = reference.mean(axis=1, skipna=False)
    keep = a.notna() & b.notna()                          # drop a race if any run timed out on it
    d = (a[keep] - b[keep]).to_numpy(dtype=float)         # paired differences, one per race
    n, delta = len(d), d.mean()
    half = T_975[n - 1] * d.std(ddof=1) / np.sqrt(n)      # t-interval half width
    return delta, delta - half, delta + half, sign_flip_p(d), n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', nargs='+', required=True, help='run folders of one method')
    parser.add_argument('--checkpoint', default='best', choices=['best', 'last', 'bc'],
                        help='method checkpoint tag (use bc for behavioral cloning)')
    parser.add_argument('--reference', nargs='+', help='run folders of another method (default: expert)')
    args = parser.parse_args()

    method = race_steps(args.method, args.checkpoint)
    reference = (race_steps(args.reference, 'best') if args.reference
                 else race_steps([EXPERT], 'team4'))
    steps = method.to_numpy(dtype=float).ravel()          # all races of all runs
    steps = steps[~np.isnan(steps)]                       # finished races only
    print(f'completion {steps.size}/{method.size}, steps {steps.mean():.2f} +/- {steps.std(ddof=1):.2f}')
    delta, low, high, p, n = compare(method, reference)
    print(f'Delta {delta:+.2f} [{low:+.2f}, {high:+.2f}], p = {p:.2g} ({n} races)')
