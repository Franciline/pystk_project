"""Run with PYTHONPATH=src python src/unit_tests/test_episode_diagnostics.py."""
from types import SimpleNamespace as NS
from tempfile import TemporaryDirectory
from pathlib import Path
import gzip,pickle
from utils.dataset_recorder import DatasetRecorder
from utils.episode_diagnostics import EpisodeDiagnostics


def env():
    kart=NS(location=[0.,0.,0.],overall_distance=0.,rescue_active=False)
    return NS(step_size=1.,kart_ix=0,world=NS(karts=[kart],time=0.))


def test_diagnostics():
    e=env(); d=EpisodeDiagnostics(e)
    for _ in range(10): d.capture()
    assert d.summary()['stuck_count']==0 # countdown excluded
    e.world.time=1;d.capture()
    for t in range(2,7):
        e.world.time=t; d.capture()
    assert d.summary()['stuck_count']==1 and d.summary()['stuck_duration_s']==5
    e.world.karts[0].rescue_active=True; d.capture(); d.capture()
    assert d.summary()['rescue_count']==1 and d.summary()['stuck_duration_s']==5
    e.world.karts[0].rescue_active=False;d.capture()
    e.world.karts[0].rescue_active=True;d.capture()
    assert d.summary()['rescue_count']==2
    e=env();del e.world.karts[0].rescue_active
    assert EpisodeDiagnostics(e).summary()['rescue_count'] is None
    # Slow progress through a window must not be mistaken for stationary time.
    e=env();d=EpisodeDiagnostics(e)
    for t in range(1,20):
        e.world.time=t;e.world.karts[0].overall_distance=t*.6
        e.world.karts[0].location=[t*.6,0,0];d.capture()
    assert d.summary()['stuck_count']==0
    with TemporaryDirectory() as directory:
        e=env();r=DatasetRecorder(e,expert_labels=False)
        r.start_episode({'v':[0]})
        e.world.time=1;r.record_step({'rescue':False},1,{'v':[1]},False,True)
        p=r.save_episode(Path(directory)/'episode.pkl.gz')
        with gzip.open(p,'rb') as f: saved=pickle.load(f)
        assert len(saved['diagnostics'])==len(saved['observations'])==2
        assert saved['expert_actions']==[None]
        assert saved['diagnostic_summary']['rescue_count']==0


if __name__=='__main__':
    test_diagnostics();print('Episode diagnostics checks passed')
