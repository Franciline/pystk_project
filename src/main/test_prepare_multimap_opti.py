"""Filesystem-only checks for extending expert references and deduplicated offline data."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main import prepare_multimap_opti as prep

ROOT=Path(__file__).resolve().parents[2]


from utils.experiment_config import load_config

def check():
    c=OmegaConf.to_container(load_config(profile='multimap25k_dagger'),resolve=True)
    assert len(c['train_tracks'])==5 and len(c['unseen_tracks'])==3 and c['N']==6
    assert set(c['train_tracks']).isdisjoint(c['unseen_tracks'])
    with TemporaryDirectory(prefix='prepare-opti-') as name:
        temp=Path(name)
        source=temp/'old_reference';source.mkdir()
        actual=ROOT/'datasets/multimap/expert_reference'
        for filename in ('summary.csv','episodes.csv','track_timeouts.json'):
            (source/filename).write_bytes((actual/filename).read_bytes())
        previous={p.name:p.read_bytes() for p in source.iterdir()}
        c['optimization']['reuse_expert']=str(source)
        c['track_timeouts']['directory']=str(temp/'reference5')
        extra=temp/'reference5_additions';extra.mkdir()
        missing=['hacienda','minigolf']
        ec=deepcopy(c);ec.update(train_tracks=missing,unseen_tracks=[])
        seeds=list(range(10000,10005))
        (extra/'config.json').write_text(json.dumps(dict(seeds=seeds,task_config=ec)))
        summary=prep.read_csv(actual/'summary.csv')[0]
        episode=prep.read_csv(actual/'episodes.csv')[0]
        prep.write_csv(extra/'summary.csv',[dict(summary,track=t,finish_steps_mean=100) for t in missing])
        prep.write_csv(extra/'episodes.csv',[dict(episode,track=t,seed=s,n_steps=100,terminated=True,truncated=False)
                                             for t in missing for s in seeds])
        # read_csv expects lower-case boolean text; Python CSV emits True, handled by .lower().
        target,limits=prep.prepare_expert(c)
        assert len(limits)==8 and limits['hacienda']==500 and limits['minigolf']==500
        assert prep.prepare_expert(c)[1]==limits
        assert previous=={p.name:p.read_bytes() for p in source.iterdir()}
        assert len(prep.read_csv(target/'summary.csv'))==8
        assert len(prep.read_csv(target/'episodes.csv'))==40
        old=temp/'offline_old';new=temp/'offline_new';old.mkdir();new.mkdir()
        original=next((ROOT/'datasets/multimap/offline/abyss').rglob('*.pkl.gz'))
        with gzip.open(original,'rb') as f: template=pickle.load(f)
        template['observations']=template['observations'][:2]
        for key in ('actions','expert_actions','rewards','terminated','truncated'):
            template[key]=template[key][:1]
        template.pop('diagnostics',None)
        for track in c['train_tracks']:
            directory=old if track in ('abyss','volcano_island','sandtrack') else new
            for epsilon in c['offline']['epsilons']:
                for seed in (0,1,2):
                    e=deepcopy(template)
                    e['metadata'].update(track=track,epsilon=epsilon,seed=seed)
                    # Identical episodes deliberately exercise exact deduplication.
                    with gzip.open(directory/f'{track}_{epsilon}_{seed}.pkl.gz','wb') as f:pickle.dump(e,f)
        merged=temp/'merged'
        prep.merge_offline(c,old,new,merged)
        manifest=json.loads((merged/'manifest.json').read_text())
        assert manifest['trajectories_before']==45 and manifest['trajectories_after']==5
        snapshot={p.relative_to(merged):p.read_bytes() for p in merged.rglob('*.pkl.gz')}
        prep.merge_offline(c,old,new,merged)
        assert snapshot=={p.relative_to(merged):p.read_bytes() for p in merged.rglob('*.pkl.gz')}
    print('PASS: eight-map reference reuses six, frozen +400 limits, five-map merge dedup, idempotency, sources preserved')


if __name__=='__main__':check()
