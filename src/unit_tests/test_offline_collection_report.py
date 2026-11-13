"""Run: PYTHONPATH=src python src/unit_tests/test_offline_collection_report.py."""
import gzip,pickle,json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from main.collect_offline import collect_dataset


class Env:
    def __init__(self,**kwargs):pass
    def close(self):pass


def test_collection():
    with TemporaryDirectory() as directory:
        root=Path(directory)
        c=dict(track='a',train_tracks=['a','b'],max_steps=5,step_size=.1,
            max_steps_by_track=dict(a=5,b=8),offline=dict(dataset=str(root/'unique'),
            raw_dataset=str(root/'raw'),epsilons=[0.,.1,.3],episodes_per_epsilon=4,seed_start=0))
        def collect(env,learner,**kw):
            assert kw['max_steps']==c['max_steps_by_track'][env.default_track]
            seed,epsilon=kw['seed'],kw['epsilon'];value=0 if epsilon==0 else epsilon+seed
            episode=dict(metadata=dict(track=env.default_track,agent='team4',seed=seed,
                epsilon=epsilon,max_steps=kw['max_steps'],step_size=.1),
                actions=[dict(steer=value)],expert_actions=[dict(steer=0)],rewards=[1],
                observations=[dict(distance_down_track=[0]),dict(distance_down_track=[value])],
                terminated=[True],truncated=[False])
            p=kw['output'];p.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(p,'wb') as f:pickle.dump(episode,f)
            return dict(n_steps=1)
        with patch('main.collect_offline.prepare_config',lambda x:x),\
             patch('main.collect_offline.STKRaceEnv',Env),\
             patch('main.collect_offline.collect_episode',collect):
            totals=collect_dataset(c)
        assert totals['trajectories_before']==24 and totals['trajectories_after']==18
        assert len(list((root/'raw').rglob('*.pkl.gz')))==24
        assert len(list((root/'unique').rglob('*.pkl.gz')))==18
        report=json.loads((root/'unique/manifest.json').read_text())
        assert len(report['retained_files'])==18
        with patch('main.collect_offline.prepare_config',lambda x:x),\
             patch('main.collect_offline.STKRaceEnv',Env),\
             patch('main.collect_offline.collect_episode',side_effect=AssertionError('Recollected')):
            assert collect_dataset(c)==totals


if __name__=='__main__':
    test_collection();print('Offline before/after deduplication and repeat-run checks passed')
