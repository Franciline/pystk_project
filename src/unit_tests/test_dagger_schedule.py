"""Run with PYTHONPATH=src python src/unit_tests/test_dagger_schedule.py."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from main import DAgger


class Env:
    def __init__(self, **kwargs): self.default_track = kwargs['track']
    def close(self): pass


def test_schedule():
    with TemporaryDirectory() as directory:
        output = Path(directory) / 'dagger'
        config = dict(N=3, T=2, initial_trajectories=1, max_steps=3, seed=0,
                      train_tracks=['abyss','minigolf'], unseen_tracks=['lighthouse'],
                      expert_benchmark=dict(runs=5,seed_start=10000), step_size=.1, output_dir=str(output),
                      evaluation=dict(seeds=[10000], min_relative_improvement=.01))
        calls, evaluations, trains = [], [], []
        def collect(env, learner, **kw):
            calls.append((kw['beta'], kw['seed'], env.default_track))
            return dict(n_steps=3)
        def train(learner, paths, iteration):
            trains.append(len(paths))
            (output / f'actor_iteration_{iteration}.pt').write_bytes(b'checkpoint')
        def evaluate(env, learner, seeds, max_steps, action):
            evaluations.append(env.default_track)
            return [dict(seed=10000,n_steps=3,terminated=True,truncated=False)], dict(
                completion_rate=1., finish_steps_mean=3., steps_mean=3.)
        with patch('main.benchmark_expert.run_benchmark', return_value={t: dict(finish_steps_mean=3.) for t in ['abyss','minigolf','lighthouse']}), patch.object(DAgger,'STKRaceEnv',Env), patch.object(DAgger,'collect_episode',collect), patch.object(DAgger,'evaluate_policy',evaluate):
            DAgger.run_dagger(config, learner=object(), train_policy=train)
        assert [c[0] for c in calls] == [1,1,0,0,0,0,0,0,0,0]
        assert [c[1] for c in calls] == list(range(10))
        assert set(c[2] for c in calls)=={'abyss','minigolf'}
        assert trains == [2,6,10]
        assert evaluations == ['abyss','minigolf','lighthouse'] * 3
        with (output/'evaluation.csv').open() as f: assert len(list(csv.DictReader(f)))==9
        assert (output/'actor_best.pt').exists()


if __name__ == '__main__':
    test_schedule()
    print('Schedule and dual-track checks passed')
