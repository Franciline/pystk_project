"""Run: PYTHONPATH=src python src/unit_tests/test_episode_deduplication.py."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import gzip
import pickle
from main.DAgger import load_initial_episodes
from utils.episode_deduplication import drop_duplicates


def test_deduplication():
    action=dict(acceleration=1,steer=0,brake=0,drift=0,fire=0,nitro=0,rescue=0)
    episode=dict(actions=[action],expert_actions=[action], rewards=[1.],
                 observations=[dict(distance_down_track=[0]),dict(distance_down_track=[1])],
                 metadata=dict(track='abyss', agent='team4'))
    duplicate=deepcopy(episode)
    duplicate['metadata']['seed']=2
    changed=deepcopy(episode)
    changed['observations'][-1]['distance_down_track']=[2]
    other_map=deepcopy(episode);other_map["metadata"]["track"]="sandtrack"
    assert len(drop_duplicates([episode,other_map]))==2
    unique=drop_duplicates([episode,duplicate,changed])
    assert len(unique)==2 and unique[0] is episode and unique[1] is changed
    with TemporaryDirectory() as directory:
        source=Path(directory)/'source'; source.mkdir()
        for i,e in enumerate([episode,duplicate,changed]):
            with gzip.open(source/f'{i}.pkl.gz','wb') as f: pickle.dump(e,f)
        paths=load_initial_episodes(None,source,Path(directory)/'dest','abyss')
        assert len(paths)==2
        with gzip.open(paths[0],'rb') as f: saved=pickle.load(f)
        assert saved['metadata']['source_file'].endswith('0.pkl.gz')
        assert saved['actions']==episode['actions']


if __name__ == '__main__':
    test_deduplication()
    print('Shared deduplication checks passed')
