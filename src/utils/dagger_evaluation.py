"""Learner-only validation; evaluation episodes never enter the training dataset."""
import csv
from pathlib import Path
import random
import numpy as np


def evaluate_policy(env, learner, seeds, max_steps, to_action, policy_factory=None):
    if not seeds or max_steps < 1:
        raise ValueError('Evaluation needs seeds and a positive step limit')
    rows = []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            env.action_space.seed(seed)
            obs, _ = env.reset(seed=seed)
            policy = policy_factory(env) if policy_factory is not None else learner
            if hasattr(policy, 'reset_episode'):
                policy.reset_episode()
            for step in range(max_steps):
                action = to_action(policy.choose_action(obs))
                obs, _, terminated, truncated, _ = env.step(action)
                truncated = bool(truncated or (step + 1 == max_steps and not terminated))
                if terminated or truncated:
                    break
            rows.append({'seed': seed, 'n_steps': step + 1,
                         'terminated': bool(terminated), 'truncated': truncated})
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    steps = [r['n_steps'] for r in rows]
    finished = [r['n_steps'] for r in rows if r['terminated']]
    summary = {
        'episodes': len(rows), 'completed': len(finished),
        'completion_rate': len(finished) / len(rows),
        # A finish on the limiting step counts as completion, even if both flags are true.
        'timeouts': sum(r['truncated'] and not r['terminated'] for r in rows),
        'steps_mean': float(np.mean(steps)), 'steps_std': float(np.std(steps, ddof=1)) if len(steps) > 1 else None,
        'finish_steps_mean': float(np.mean(finished)) if finished else None,
        'finish_steps_std': float(np.std(finished, ddof=1)) if len(finished) > 1 else None,
    }
    return rows, summary


def is_improvement(current, best, min_steps):
    if best is None:
        return True
    if current['completion_rate'] != best['completion_rate']:
        return current['completion_rate'] > best['completion_rate']
    if current['finish_steps_mean'] is None:
        return False
    gain = best['finish_steps_mean'] - current['finish_steps_mean']
    return gain > 0 and gain >= min_steps


def append_metrics(path, rows):
    path = Path(path)
    new = not path.exists()
    with path.open('a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)
