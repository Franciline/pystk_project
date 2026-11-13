"""Overlay validation-selected multimap policies against the saved single-map expert lap."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.dagger_actor import ActorPolicy
from main.compare_generalization import load_rlpd
from main.compare_trajectories import collect,plot_trajectories
from pystk2_gymnasium.envs import STKRaceEnv,AgentSpec

ROOT=Path(__file__).resolve().parents[2]
POLICIES=[('Behavioral Cloning','bc'),('DAgger','dagger'),('RLPD','rlpd')]


def selection(root):
    chosen={}
    for method,filename in POLICIES:
        candidates=[]
        for seed in (1000,2000,3000):
            directory=root/f'{"rlpd" if method=="RLPD" else "dagger"}_seed{seed}'
            if method=='Behavioral Cloning':
                validation=pd.read_csv(directory/'evaluation.csv')
                row=validation[validation.iteration==1].iloc[0].to_dict()
                checkpoint=directory/'actor_iteration_1.pt'
            else:
                row=json.loads((directory/'best_evaluation.json').read_text())
                checkpoint=directory/('actor_best.pt' if method=='DAgger' else 'agent_best.pt')
            finish=row['finish_steps_mean']
            score=(-float(row['completion_rate']),float(finish) if finish is not None and pd.notna(finish) else float('inf'),seed)
            candidates.append((score,dict(method=method,training_seed=seed,checkpoint=str(checkpoint.resolve()),validation=row)))
        _,best=min(candidates,key=lambda item:item[0])
        best['sha256']=hashlib.sha256(Path(best['checkpoint']).read_bytes()).hexdigest()
        chosen[filename]=best
    return chosen


def metrics(directory,output):
    config=json.loads((directory/'config.json').read_text())
    summary=pd.read_csv(directory/'summary.csv').set_index('model')
    single=Path(config['expert_reference'])
    data={}
    # Use one common grid for the new plot and a directly comparable single-map table.
    for regime,source in [('Multimap',directory),('Single map',single)]:
        for label,name in [('Expert (team4)','expert'),*POLICIES]:
            d=pd.read_csv(source/f'{name}.csv')
            d=d[d.progress_m>=0].copy()
            if not np.isfinite(d[['progress_m','x','z']].to_numpy()).all():raise ValueError('Non-finite trajectory')
            if (np.diff(d.progress_m)<0).any():raise ValueError('Backtracking needs a separate alignment rule')
            d=d.drop_duplicates('progress_m',keep='first')
            data[regime,name]=d
    start=max(d.progress_m.iloc[0] for d in data.values())
    end=min(d.progress_m.iloc[-1] for d in data.values())
    if not end>start:raise ValueError('No common track-progress range')
    grid=np.linspace(start,end,int(np.ceil((end-start)/.1))+1)
    def aligned(d):return np.column_stack([np.interp(grid,d.progress_m,d[axis]) for axis in ('x','z')])
    expert=aligned(data['Multimap','expert'])
    rows=[];profiles=dict(progress_m=grid)
    for regime,source in [('Multimap',directory),('Single map',single)]:
        details=pd.read_csv(source/'summary.csv').set_index('model')
        for label,name in [('Expert (team4)','expert'),*POLICIES]:
            error=np.linalg.norm(aligned(data[regime,name])-expert,axis=1)
            row=details.loc[label]
            if str(row.terminated).lower()!='true':raise ValueError('Incomplete trajectory; distance needs coverage reporting')
            mean=float(np.trapezoid(error,grid)/(end-start))
            p95=float(np.percentile(error,95))
            if name=='expert':assert mean==p95==0
            rows.append(dict(regime=regime,model=label,mean_distance_to_expert_m=mean,
                p95_distance_to_expert_m=p95,lap_steps=int(row.n_steps)))
            profiles[f'{regime.lower().replace(" ","_")}_{name}_distance_m']=error
    table=pd.DataFrame(rows)
    stem=output/config['plot_stem']
    table[table.regime=='Multimap'].to_csv(stem.with_name(stem.name+'_metrics.csv'),index=False)
    table.to_csv(stem.with_name(stem.name+'_comparison.csv'),index=False)
    pd.DataFrame(profiles).to_csv(stem.with_name(stem.name+'_distance_profile.csv'),index=False)
    stem.with_name(stem.name+'_metrics.json').write_text(json.dumps(dict(source=str(directory.resolve()),
        expert_reference=str(single.resolve()),alignment='Equal track progress, linear interpolation, original unscaled world X/Z',
        progress_start_m=float(start),progress_end_m=float(end),samples=len(grid),spacing_m=float(grid[1]-grid[0]),
        mean='Trapezoidal distance integral / progress span',p95='95th percentile of distance samples',
        note='Common overlap grid includes both single-map and multimap runs; negative startup sentinel excluded',
        selection=config['selection']),indent=2))
    print(table.to_string(index=False))


def run(args):
    directory=args.data_dir;directory.mkdir(parents=True,exist_ok=True)
    if not args.plot_only:
        chosen=selection(args.root)
        reference=json.loads((args.expert_reference/'config.json').read_text())
        config=dict(track=reference['track'],seed=reference['seed'],step_size=reference['step_size'],max_steps=reference['max_steps'],
            expert_reference=str(args.expert_reference.resolve()),selection=chosen,
            selection_rule='Training-map validation completion then macro finishing steps; test/unseen excluded',
            plot_stem='multimap_trajectories_abyss_seed20000',
            plot_title='Expert, Behavioral Cloning, DAgger and RLPD trajectories on Abyss map, seed 20000\nPolicies trained on five maps')
        path=directory/'config.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Saved selection/config differs; use a fresh data directory')
        path.write_text(json.dumps(config,indent=2))
        for filename in ('expert.csv','expert_items.csv','track.npz'):
            source=args.expert_reference/filename;destination=directory/filename
            if destination.exists():
                if destination.read_bytes()!=source.read_bytes():raise ValueError(f'Expert reference differs: {filename}')
            else:shutil.copy2(source,destination)
        if not (directory/'summary.csv').exists():
            old=pd.read_csv(args.expert_reference/'summary.csv')
            old[old.model=='Expert (team4)'].to_csv(directory/'summary.csv',index=False)
        env=STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track=config['track'],num_kart=1,laps=1,step_size=config['step_size'])
        try:
            for method,name in POLICIES:
                if (directory/f'{name}.csv').exists():continue
                checkpoint=Path(chosen[name]['checkpoint'])
                policy=load_rlpd(checkpoint,args.device) if method=='RLPD' else ActorPolicy.load(checkpoint,device=args.device)
                detail=collect(env,policy,config['seed'],config['max_steps'],directory/f'{name}.csv')
                pd.DataFrame([dict(model=method,**detail)]).to_csv(directory/'summary.csv',mode='a',header=False,index=False)
                print(method,'training seed',chosen[name]['training_seed'],detail,flush=True)
        finally:env.close()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plot_trajectories(directory,args.output_dir,width_scale=4)
    metrics(directory,args.output_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'datasets/multimap/full5maps50k')
    parser.add_argument('--expert-reference',type=Path,default=ROOT/'datasets/protocol/trajectories_abyss_seed20000_normalized')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'datasets/multimap/full5maps50k/trajectories_abyss_seed20000')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'datasets/plots')
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--plot-only',action='store_true')
    run(parser.parse_args())
