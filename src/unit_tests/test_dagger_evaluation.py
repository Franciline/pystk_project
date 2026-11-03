"""Run with PYTHONPATH=src python src/unit_tests/test_dagger_evaluation.py."""
from utils.dagger_evaluation import evaluate_policy, is_improvement


class Env:
    action_space = type('Space', (), {'seed': lambda self, seed: None})()

    def reset(self, seed):
        self.seed, self.steps = seed, 0
        return {}, {}

    def step(self, action):
        self.steps += 1
        # Seed 1 finishes early; seed 2 at limit with both flags; seed 3 times out.
        finish = self.seed == 1 or (self.seed == 2 and self.steps == 3)
        return {}, 0, finish, self.seed == 2 and finish, {}


class Learner:
    def choose_action(self, obs):
        return {}


def test_evaluation():
    rows, summary = evaluate_policy(Env(), Learner(), [1, 2, 3], 3, lambda x: x)
    assert [r['n_steps'] for r in rows] == [1, 3, 3]
    assert summary['completed'] == 2 and summary['timeouts'] == 1
    assert summary['finish_steps_mean'] == 2 and summary['steps_mean'] == 7 / 3
    assert is_improvement(summary, None, 1)
    assert not is_improvement(summary, summary, 1)
    faster = dict(summary, finish_steps_mean=1.5)
    assert not is_improvement(faster, summary, 1)
    assert is_improvement(dict(summary, finish_steps_mean=1), summary, 1)
    assert is_improvement(dict(summary, completion_rate=1), summary, 1)
    _, failed = evaluate_policy(Env(), Learner(), [3], 3, lambda x: x)
    assert failed['finish_steps_mean'] is None and failed['steps_std'] is None
    assert not is_improvement(failed, failed, 1)


if __name__ == '__main__':
    test_evaluation()
    print('Evaluation checks passed')
