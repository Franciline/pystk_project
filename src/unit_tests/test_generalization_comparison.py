"""Run: PYTHONPATH=src python src/unit_tests/test_generalization_comparison.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd
from main import compare_generalization as module


class Env:
    def __init__(self, **kwargs): self.default_track = kwargs['track']; self.closed = False
    def close(self): self.closed = True


def test_comparison():
    calls=[]
    def evaluate(env, policy, seeds, steps, action):
        calls.append((env.default_track, len(seeds)))
        rows=[dict(seed=s,n_steps=500+i,terminated=True,truncated=False) for i,s in enumerate(seeds)]
        return rows,dict(episodes=5,completed=5,completion_rate=1.,timeouts=0,steps_mean=502.,steps_std=1.,finish_steps_mean=502.,finish_steps_std=1.)
    config=dict(comparison=dict(tracks=['abyss','snowtuxpeak'],seeds=list(range(5))),
                max_steps=1000,step_size=.1)
    with TemporaryDirectory() as directory:
        output=Path(directory)
        with patch.object(module.ActorPolicy,'load',return_value=object()), patch.object(module,'load_rlpd',return_value=object()), patch.object(module,'STKRaceEnv',Env), patch.object(module,'evaluate_policy',evaluate):
            module.compare(config,Path('actor_best.pt'),Path('agent_best.pt'),'cpu',output)
        assert calls==[('abyss',5),('abyss',5),('snowtuxpeak',5),('snowtuxpeak',5)]
        assert len(pd.read_csv(output/'episodes.csv'))==20
        assert len(pd.read_csv(output/'summary.csv'))==4
        assert (output/'generalization_boxplots.svg').stat().st_size>0
        assert not list(output.glob('*.pdf')) and not list(output.glob('*.png'))


if __name__ == '__main__':
    test_comparison()
    print('Generalization comparison checks passed')
