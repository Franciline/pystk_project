"""Collect shared raw Team4 episodes, retain unique episodes, and report actual counts."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import pickle
import shutil
import sys
from omegaconf import OmegaConf

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main.DAgger import collect_episode
from pystk2_gymnasium.envs import STKRaceEnv,AgentSpec
from utils.multitrack import train_tracks,track_step_limit,prepare_config
from utils.episode_deduplication import drop_duplicates,deduplication_report

ROOT=Path(__file__).resolve().parents[2]


def dataset_path(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT/path


def collect_dataset(config):
    c=prepare_config(config)
    tracks=train_tracks(c);offline=c['offline'];target=dataset_path(offline['dataset'])
    raw=dataset_path(offline.get('raw_dataset',str(target)+'_raw'))
    if raw==target or raw in target.parents or target in raw.parents:
        raise ValueError('Raw and deduplicated datasets must be separate, non-nested directories')
    signature=dict(tracks=tracks,epsilons=list(offline['epsilons']),
        seed_start=offline['seed_start'],episodes_per_epsilon=offline['episodes_per_epsilon'],
        step_size=c['step_size'],max_steps_by_track={t:track_step_limit(c,t) for t in tracks})
    manifest_path=target/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['collection']!=signature:
        raise ValueError(f'Collection settings differ at {target}; use a new dataset directory')
    env=STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track=tracks[0],
                   num_kart=1,laps=1,step_size=c['step_size'])
    episodes=[];sources={}
    try:
        for track in tracks:
            env.default_track=track
            for epsilon in offline['epsilons']:
                for seed in range(offline['seed_start'],offline['seed_start']+offline['episodes_per_epsilon']):
                    output=raw/track/f'team4_eps{float(epsilon)}_seed{seed}.pkl.gz'
                    if not output.exists():
                        metrics=collect_episode(env,None,beta=1,seed=seed,max_steps=track_step_limit(c,track),
                            output=output,iteration=0,step_size=c['step_size'],epsilon=epsilon,
                            diagnostics=c.get('diagnostics'))
                        print(f'{track}, epsilon={epsilon}, seed={seed}: {metrics["n_steps"]} steps',flush=True)
                    with gzip.open(output,'rb') as file:
                        episode=pickle.load(file)
                    meta=episode['metadata']
                    if any(meta.get(key)!=value for key,value in dict(track=track,epsilon=epsilon,
                        seed=seed,step_size=c['step_size'],max_steps=track_step_limit(c,track)).items()):
                        raise ValueError(f'Existing raw episode has different collection settings: {output}')
                    episodes.append(episode);sources[id(episode)]=output
    finally:
        env.close()
    unique=drop_duplicates(episodes)
    report=deduplication_report(episodes,unique)
    target.mkdir(parents=True,exist_ok=True)
    kept=[]
    for episode in unique:
        source=sources[id(episode)];relative=source.relative_to(raw);destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if destination.read_bytes()!=source.read_bytes():
                raise ValueError(f'Existing retained episode differs: {destination}')
        else:
            shutil.copyfile(source,destination)
        kept.append(str(relative))
    unexpected={str(p.relative_to(target)) for p in target.rglob('*.pkl.gz')}-set(kept)
    if unexpected:
        raise ValueError(f'Extra episode files in retained dataset: {sorted(unexpected)}; use a new directory')
    with (target/'deduplication_summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    by_track=[]
    for track in tracks:
        groups=[row for row in report if row['track']==track]
        by_track.append(dict(track=track,**{key:sum(row[key] for row in groups)
            for key in ('trajectories_before','trajectories_after','steps_before','steps_after')}))
    with (target/'deduplication_by_track.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(by_track[0]));writer.writeheader();writer.writerows(by_track)
    totals=dict(trajectories_before=len(episodes),trajectories_after=len(unique),
                steps_before=sum(len(e['actions']) for e in episodes),
                steps_after=sum(len(e['actions']) for e in unique))
    manifest_path.write_text(json.dumps(dict(collection=signature,raw_dataset=str(raw),
        retained_files=kept,totals=totals),indent=2))
    for row in report:
        print(f"{row['track']} epsilon={row['epsilon']}: "
              f"{row['trajectories_before']} -> {row['trajectories_after']} trajectories, "
              f"{row['steps_before']} -> {row['steps_after']} steps",flush=True)
    for row in by_track:
        print(f"{row['track']} total: {row['trajectories_before']} -> {row['trajectories_after']} trajectories",flush=True)
    print(f"TOTAL after deduplication: {len(episodes)} -> {len(unique)} trajectories; "
          f"{totals['steps_after']} transitions retained. Report: {target/'deduplication_summary.csv'}",flush=True)
    return totals


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('rlpd_config.yaml'))
    parser.add_argument('--experiment',type=Path)
    args,overrides=parser.parse_known_args()
    c=OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(args.config),
        OmegaConf.load(args.experiment) if args.experiment else {},OmegaConf.from_dotlist(overrides)),resolve=True)
    collect_dataset(c)
