"""Run with PYTHONPATH=src python src/unit_tests/test_multitrack.py."""
from unittest.mock import patch
from utils.multitrack import train_tracks,evaluation_tracks,next_track,evaluate_tracks


def test_maps():
    c=dict(track='old',train_tracks=['a','b'],unseen_tracks=['new'],max_steps=1,
           evaluation=dict(seeds=[1]))
    assert train_tracks(dict(track='old'))==['old']
    assert evaluation_tracks(c)==['a','b','new']
    assert next_track(['a','b'],{'a':1000,'b':500})=='b'
    class Env: pass
    def evaluate(env,*a,**kw):
        mean={'a':100.,'b':200.,'new':1.}[env.default_track]
        return [dict(seed=1,n_steps=mean,terminated=True,truncated=False)],dict(
            episodes=1,completed=1,completion_rate=1.,timeouts=0,steps_mean=mean,
            finish_steps_mean=mean,finish_steps_std=None,steps_std=None)
    with patch('utils.multitrack.evaluate_policy',evaluate):
        rows,summaries,score=evaluate_tracks(Env(),None,c,'unused',lambda x:x)
    assert len(rows)==len(summaries)==3
    assert score['finish_steps_mean']==150. # unseen value must not affect selection
    assert score['episodes']==2


if __name__=='__main__':
    test_maps();print('Multi-track checks passed')
