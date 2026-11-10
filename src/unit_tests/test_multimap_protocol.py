"""Run: PYTHONPATH=src python src/unit_tests/test_multimap_protocol.py."""
import csv,json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from main.benchmark_expert import prepare_protocol
from utils.multitrack import MapCollection,prepare_config,track_step_limit,save_expert_reference


def test_protocol():
    with TemporaryDirectory() as directory:
        config=dict(train_tracks=['a','b'],unseen_tracks=['c'],max_steps=10,step_size=.1,
            expert_benchmark=dict(seed_start=10,runs=2),track_timeouts=dict(mode='expert',
                directory=str(Path(directory)/'reference'),margin_steps=400,benchmark_max_steps=5000))
        def benchmark(c,folder,seeds):
            assert c['max_steps']==5000 and c['max_steps_by_track']=={}
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'config.json').write_text(json.dumps(dict(task_config=c,seeds=seeds)))
            with (folder/'episodes.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['track','seed','n_steps','terminated'])
                writer.writeheader()
                for track,steps in [('a',[100,120]),('b',[50,60]),('c',[200,205])]:
                    for seed,n in zip(seeds,steps):writer.writerow(dict(track=track,seed=seed,n_steps=n,terminated=True))
            (folder/'summary.csv').write_text('track\na\nb\nc\n')
        with patch('main.benchmark_expert.run_benchmark',benchmark):
            reference,limits=prepare_protocol(config)
        assert limits==dict(a=520,b=460,c=605)
        changed=deepcopy(config);changed['seed']=3000
        with patch('main.benchmark_expert.run_benchmark',side_effect=AssertionError('Recalibrated')):
            loaded=prepare_config(changed)
        assert loaded['max_steps_by_track']==limits
        assert track_step_limit(loaded,'c')==605
        assert 'max_steps_by_track' not in config
        save_expert_reference(loaded,Path(directory)/'run1')
        save_expert_reference(loaded,Path(directory)/'run2')
        assert (Path(directory)/'run1/expert_benchmark/track_timeouts.json').read_bytes()==(Path(directory)/'run2/expert_benchmark/track_timeouts.json').read_bytes()
        changed['track_timeouts']['margin_steps']=401
        try:prepare_protocol(changed)
        except ValueError:pass
        else:raise AssertionError('Changed frozen protocol accepted')
    counts=MapCollection(['a','b'],1000)
    assert counts.next_seed('a')==counts.next_seed('b')==1000
    counts.record('a',100);assert counts.next_seed('a')==1001 and counts.next_seed('b')==1000
    counts.record('b',50)
    rows=counts.summary(dict(a=1,b=1),dict(a=100,b=50),iteration=2)
    assert sum(r['cumulative_steps'] for r in rows)==150
    assert sum(r['cumulative_step_fraction'] for r in rows)==1


if __name__=='__main__':
    test_protocol();print('Frozen timeout and per-map seed checks passed')
