"""Run with PYTHONPATH=src python src/unit_tests/test_dagger_schedule.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import csv
from main import DAgger


class Env:
    def __init__(self, **kwargs): pass
    def close(self): pass


def test_schedule():
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config = dict(N=3, steps_per_iteration=5, max_steps=3, seed=1,
            online_seed_start=1000, online_epsilon=.1, track='abyss', step_size=.1,
            output_dir=str(output), initialization=dict(mode='generate_offline',
                epsilons=[0,.1,.3], episodes_per_epsilon=3),
            evaluation=dict(seeds=[10000], min_steps_improvement=0))
        calls, trained = [], []
        def collect(env, learner, **kw):
            calls.append(kw)
            return dict(iteration=kw['iteration'], n_steps=3)
        def train(learner, paths, iteration):
            trained.append(len(paths))
            (output / f'actor_iteration_{iteration}.pt').write_bytes(str(iteration).encode())
        summary = dict(completion_rate=1., finish_steps_mean=3.)
        with patch.object(DAgger, 'STKRaceEnv', Env), patch.object(DAgger, 'collect_episode', collect), patch.object(DAgger, 'evaluate_policy', return_value=([dict(seed=10000,n_steps=3,terminated=True,truncated=False)], summary)):
            DAgger.run_dagger(config, learner=object(), train_policy=train)
        assert len(calls)==13 and trained==[9,11,13]
        assert [c['epsilon'] for c in calls[:9]]==[0]*3+[.1]*3+[.3]*3
        assert all(c['beta']==0 and c['epsilon']==.1 for c in calls[9:])
        assert [c['seed'] for c in calls]==list(range(1000,1013))
        assert (output/'actor_best.pt').read_bytes()==b'1'
        assert (output/'actor_last.pt').read_bytes()==b'3'
        with (output/'evaluation.csv').open() as file:
            rows=list(csv.DictReader(file))
        assert [int(r['online_steps']) for r in rows]==[0,6,12]
        assert all(int(r['initial_steps'])==27 for r in rows)
    action = dict(acceleration=1,steer=0,brake=0,drift=0,fire=0,nitro=0,rescue=1)
    assert DAgger.to_action(action)['rescue'] is False


if __name__ == '__main__':
    test_schedule()
    print('Step-budget, epsilon, and best/last checkpoint checks passed')
