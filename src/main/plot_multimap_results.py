"""Evaluate BC/DAgger/RLPD across seeds and maps; plot final results and aligned learning curves."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT=Path(__file__).resolve().parents[2]
METHODS=('Behavioral Cloning','DAgger','RLPD')
COLORS={'Behavioral Cloning':'#55A868','DAgger':'#4C72B0','RLPD':'#DD8452'}
LABELS={'abyss':'Abyss','volcano_island':'Volcan Island','sandtrack':'Shifting Sands','hacienda':'Hacienda',
        'minigolf':'Minigolf','snowtuxpeak':'Snow Peak','xr591':'XR591','zengarden':'Zen Garden'}
MARKERS=['o','s','^']
_env=None


def initialize_worker():
    import torch
    torch.set_num_threads(1)


def evaluate_job(job):
    global _env
    from multiprocessing.util import Finalize
    from agents.dagger_actor import ActorPolicy
    from agents.team4.agent4 import Agent4
    from main.DAgger import to_action
    from pystk2_gymnasium.envs import STKRaceEnv,AgentSpec
    from utils.dagger_evaluation import evaluate_policy
    from utils.rlpd_evaluation_opti import load_policy
    config=job['config'];signature=job['signature'];track=signature['track'];method=signature['method']
    if _env is None:
        _env=STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track=track,num_kart=1,laps=1,
                       step_size=config['step_size'],use_subprocess=True)
        Finalize(None,_env.close,exitpriority=10)
    _env.default_track=track
    if method=='Expert (team4)':
        policy=None
    elif method=='RLPD':
        policy=load_policy(Path(job['checkpoint']),config)
    else:
        policy=ActorPolicy.load(job['checkpoint'],device='cpu')
    rows,summary=evaluate_policy(_env,policy,signature['race_seeds'],signature['max_steps'],to_action,
        policy_factory=(lambda env:Agent4(env,path_lookahead=3)) if policy is None else None,
        diagnostics=config.get('diagnostics'))
    return rows,summary


def read_runs(root,seeds):
    runs={};reference=None
    for method,prefix in (('DAgger','dagger'),('RLPD','rlpd')):
        for seed in seeds:
            directory=root/f'{prefix}_seed{seed}'
            config=json.loads((directory/'config.json').read_text())
            if reference is None:reference=config
            for key in ('train_tracks','unseen_tracks','step_size','max_steps_by_track','evaluation'):
                if config[key]!=reference[key]:raise ValueError(f'Protocol differs for {directory}: {key}')
            runs[method,seed]=(directory,config)
    return runs,reference


def aligned_history(runs,step=5000,total=50000):
    tables=[]
    for (method,seed),(directory,config) in runs.items():
        data=pd.read_csv(directory/'evaluation_by_track.csv')
        tracks=config['train_tracks']+config['unseen_tracks']
        if set(data.track)!=set(tracks):raise ValueError(f'Unexpected evaluation maps at {directory}')
        data['actual_online_steps']=data.online_steps if method=='DAgger' else data.env_steps
        data['target_online_steps']=((data.iteration-1)*step if method=='DAgger' else
                                     (data.env_steps//step)*step).astype(int)
        if data.duplicated(['target_online_steps','track']).any():raise ValueError(f'Duplicate target checkpoints at {directory}')
        expected=list(range(0 if method=='DAgger' else step,total+1,step))
        unexpected=set(data.target_online_steps)-set(expected)
        if unexpected:raise ValueError(f'Unexpected target checkpoints: {unexpected}')
        # Insert missing checkpoints, rather than relabel later observations or interpolate.
        index=pd.MultiIndex.from_product([expected,tracks],names=['target_online_steps','track'])
        data=data.set_index(['target_online_steps','track']).reindex(index).reset_index()
        data['method']=method;data['training_seed']=seed
        data['split']=np.where(data.track.isin(config['train_tracks']),'train','unseen')
        tables.append(data)
    return pd.concat(tables,ignore_index=True)


def make_jobs(root,runs,reference,seeds):
    tracks=reference['train_tracks']+reference['unseen_tracks'];jobs=[]
    final=root/'final_test20_all_seeds'
    for method in (*METHODS,'Expert (team4)'):
        for seed in (seeds if method!='Expert (team4)' else [None]):
            if method=='Expert (team4)':
                directory,config=next(iter(runs.values()));checkpoint=None;digest=None;slug='expert'
            else:
                parent='DAgger' if method=='Behavioral Cloning' else method
                directory,config=runs[parent,seed]
                checkpoint=directory/('actor_iteration_1.pt' if method=='Behavioral Cloning' else 'actor_best.pt' if method=='DAgger' else 'agent_best.pt')
                digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                slug={'Behavioral Cloning':'bc','DAgger':'dagger','RLPD':'rlpd'}[method]+f'_seed{seed}'
            for track in tracks:
                signature=dict(method=method,training_seed=seed,track=track,race_seeds=list(range(20000,20020)),
                    checkpoint_sha256=digest,step_size=config['step_size'],max_steps=config['max_steps_by_track'][track],
                    encoder=config['encoder'],normalize_observations=config.get('normalize_observations'),device='cpu')
                c=deepcopy(config)
                jobs.append(dict(signature=signature,config=c,checkpoint=str(checkpoint.resolve()) if checkpoint else None,
                                 output=str(final/slug/track)))
    return jobs


def write_job(job,rows,summary):
    destination=Path(job['output']);destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='writing-',dir=destination.parent) as temporary:
        stage=Path(temporary)/'result';stage.mkdir()
        pd.DataFrame(rows).to_csv(stage/'episodes.csv',index=False)
        (stage/'summary.json').write_text(json.dumps(summary,indent=2))
        (stage/'config.json').write_text(json.dumps(dict(signature=job['signature'],checkpoint=job['checkpoint']),indent=2))
        stage.rename(destination)


def completed_job(job):
    path=Path(job['output'])
    if not path.exists():return False
    previous=json.loads((path/'config.json').read_text())['signature']
    if previous!=job['signature']:raise ValueError(f'Existing test protocol/checkpoint differs: {path}')
    data=pd.read_csv(path/'episodes.csv')
    if sorted(data.seed)!=job['signature']['race_seeds']:raise ValueError(f'Incomplete saved test: {path}')
    return True


def reuse_expert(job):
    if job['signature']['method']!='Expert (team4)':return False
    source=ROOT/'datasets/multimap/comparison20/expert'
    if not (source/'episodes.csv').exists():return False
    previous=json.loads((source/'config.json').read_text())
    c=previous['task_config'];signature=job['signature'];track=signature['track']
    if previous['seeds']!=signature['race_seeds'] or c['step_size']!=signature['step_size']:return False
    if c.get('max_steps_by_track',{}).get(track)!=signature['max_steps']:return False
    rows=pd.read_csv(source/'episodes.csv');rows=rows[rows.track==track].drop(columns='track')
    summaries=pd.read_csv(source/'summary.csv');summary=summaries[summaries.track==track]
    if sorted(rows.seed)!=signature['race_seeds'] or len(summary)!=1:return False
    values=summary.drop(columns='track').iloc[0].to_dict()
    values={key:None if pd.isna(value) else value for key,value in values.items()}
    write_job(job,rows.to_dict('records'),values)
    return True


def evaluate_all(jobs,workers):
    pending=[]
    for job in jobs:
        if not completed_job(job) and not reuse_expert(job):pending.append(job)
    print(f'{len(jobs)-len(pending)} model/map tests reused; {len(pending)} tests to run (20 laps each)',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=initialize_worker) as pool:
        futures={pool.submit(evaluate_job,job):job for job in pending}
        for done,future in enumerate(as_completed(futures),1):
            job=futures[future];rows,summary=future.result()
            write_job(job,rows,summary)
            s=job['signature']
            print(f"[{done}/{len(pending)}] {s['method']} seed {s['training_seed']} - {s['track']}: {summary['completed']}/20",flush=True)


def final_tables(jobs,reference):
    rows=[];summaries=[]
    for job in jobs:
        if not completed_job(job):raise ValueError(f"Missing final test: {job['output']}")
        s=job['signature'];path=Path(job['output'])
        tags=dict(method=s['method'],training_seed=s['training_seed'],track=s['track'],
                  split='train' if s['track'] in reference['train_tracks'] else 'unseen')
        d=pd.read_csv(path/'episodes.csv')
        d['terminated']=d.terminated.astype(str).str.lower().eq('true')
        for key,value in tags.items():d[key]=value
        rows.append(d)
        summaries.append(dict(**tags,**json.loads((path/'summary.json').read_text())))
    return pd.concat(rows,ignore_index=True),pd.DataFrame(summaries)


def macro_history(history,reference):
    records=[]
    for (method,seed,target,split),data in history.groupby(['method','training_seed','target_online_steps','split'],sort=False):
        required=len(reference['train_tracks'] if split=='train' else reference['unseen_tracks'])
        value=float(data.completion_rate.mean()) if len(data)==required and data.completion_rate.notna().all() else np.nan
        records.append(dict(method=method,training_seed=seed,target_online_steps=target,split=split,completion_rate=value,
            actual_online_steps=float(data.actual_online_steps.iloc[0]) if data.actual_online_steps.notna().all() else np.nan))
    return pd.DataFrame(records)


def plot_evolution(history,reference,output,seeds):
    macro=macro_history(history,reference);statistics=[]
    expert_path=output/'multimap_expert_summary.csv'
    if not expert_path.exists():
        expert_path=Path(reference['expert_reference_dir'])/'summary.csv'
    expert=pd.read_csv(expert_path).set_index('track')
    expert_baselines=[]
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.6))
    for ax,split in zip(axes,('train','unseen')):
        for method in ('DAgger','RLPD'):
            d=macro[(macro.method==method)&(macro.split==split)]
            agg=d.groupby('target_online_steps').completion_rate.agg(['mean','std','count']).reset_index()
            agg.loc[agg['count']!=len(seeds),['mean','std']]=np.nan
            x=agg.target_online_steps.to_numpy();mean=agg['mean'].to_numpy()*100;std=agg['std'].to_numpy()*100
            ax.plot(x,mean,color=COLORS[method],marker='o',markersize=4,label=method)
            ax.fill_between(x,np.maximum(0,mean-std),np.minimum(100,mean+std),color=COLORS[method],alpha=.16)
            agg['method']=method;agg['split']=split;statistics.append(agg)
        baseline=macro[(macro.method=='DAgger')&(macro.target_online_steps==0)&(macro.split==split)].completion_rate
        ax.axhline(baseline.mean()*100,color=COLORS['Behavioral Cloning'],linestyle='--',label='Behavioral Cloning')
        ax.axhspan(max(0,(baseline.mean()-baseline.std(ddof=1))*100),min(100,(baseline.mean()+baseline.std(ddof=1))*100),
                   color=COLORS['Behavioral Cloning'],alpha=.08)
        maps=reference['train_tracks'] if split=='train' else reference['unseen_tracks']
        expert_mean=float(expert.loc[maps,'completion_rate'].mean())
        ax.axhline(expert_mean*100,color='#333333',linestyle=':',linewidth=1.6,zorder=1.5,label='Expert (team4)')
        expert_baselines.append(dict(split=split,maps=len(maps),completion_rate=expert_mean,
            episodes=int(expert.loc[maps,'episodes'].sum()),completed=int(expert.loc[maps,'completed'].sum())))
        panel_title=f'Training maps ({len(maps)})' if split=='train' else f'Unseen maps ({len(maps)})'
        ax.set(title=panel_title,xlabel='Target online steps',ylabel='Completion rate (%)',ylim=(-3,103))
        ax.set_xticks(np.arange(0,50001,5000));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:g}k'))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,frameon=False)
    fig.suptitle('Completion during training - mean ± std across training seeds',y=.99)
    fig.subplots_adjust(left=.055,right=.99,bottom=.20,top=.86,wspace=.16);fig.savefig(output/'multimap_completion_evolution.svg',bbox_inches='tight');plt.close(fig)
    pd.DataFrame(expert_baselines).to_csv(output/'multimap_evolution_expert_baseline.csv',index=False)
    macro.to_csv(output/'multimap_evolution_by_seed.csv',index=False)
    pd.concat(statistics,ignore_index=True).to_csv(output/'multimap_evolution_summary.csv',index=False)


def plot_final(episodes,summary,reference,output,seeds):
    tracks=reference['train_tracks']+reference['unseen_tracks']
    final=summary[summary.method.isin(METHODS)].copy()
    expert=summary[summary.method=='Expert (team4)'].set_index('track').finish_steps_mean
    final['expert_mean_steps']=final.track.map(expert)
    final['relative_steps']=final.finish_steps_mean/final.expert_mean_steps
    final.to_csv(output/'multimap_final_summary_by_seed.csv',index=False)
    summary[summary.method=='Expert (team4)'].to_csv(output/'multimap_expert_summary.csv',index=False)
    episodes.to_csv(output/'multimap_final_episodes.csv',index=False)
    # Per-map completion: each point is a trained model, not an individual lap.
    fig,axes=plt.subplots(1,2,figsize=(14,4.7),gridspec_kw={'width_ratios':[5,3]})
    for ax,split in zip(axes,('train','unseen')):
        current=reference['train_tracks'] if split=='train' else reference['unseen_tracks']
        for m,method in enumerate(METHODS):
            for i,track in enumerate(current):
                group=final[(final.method==method)&(final.track==track)].set_index('training_seed').reindex(seeds)
                values=group.completion_rate.to_numpy()*100;x=i+(m-1)*.23
                mean=float(values.mean());std=float(values.std(ddof=1))
                errors=np.array([[min(std,mean)],[min(std,100-mean)]])
                ax.errorbar(x,mean,yerr=errors,color=COLORS[method],fmt='_',capsize=4,markersize=13,zorder=2)
                for j,value in enumerate(values):ax.scatter(x+(j-1)*.035,value,marker=MARKERS[j],color=COLORS[method],s=36,zorder=3)
        ax.set_xticks(range(len(current)),[LABELS[t] for t in current],rotation=20,ha='right')
        ax.set(title='Training maps' if split=='train' else 'Unseen maps',ylabel='Completion rate (%)',ylim=(-8,118))
    handles=[Line2D([],[],color=COLORS[m],marker='o',linestyle='',label=m) for m in METHODS]
    handles += [Line2D([],[],color='gray',marker=MARKERS[i],linestyle='',label=f'Seed {seed}') for i,seed in enumerate(seeds)]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=9)
    fig.suptitle('Final-test completion - mean ± std across 3 training seeds')
    fig.tight_layout(rect=(0,.10,1,.94));fig.savefig(output/'multimap_completion.svg',bbox_inches='tight');plt.close(fig)
    # One independent y-axis per map; timeouts remain visible.
    fig,axes=plt.subplots(1,len(tracks),figsize=(3.5*len(tracks),4.8),squeeze=False)
    for ax,track in zip(axes[0],tracks):
        for i,method in enumerate(METHODS):
            data=episodes[(episodes.method==method)&(episodes.track==track)]
            box=ax.boxplot([data.n_steps],positions=[i],widths=.34,patch_artist=True,showfliers=False,manage_ticks=False,
                medianprops=dict(color='#333333',linewidth=1.6))
            box['boxes'][0].set(facecolor=COLORS[method],alpha=.35)
            for j,seed in enumerate(seeds):
                d=data[data.training_seed==seed].sort_values('seed');x=i+(j-1)*.075+np.linspace(-.018,.018,len(d))
                success=d.terminated.to_numpy();values=d.n_steps.to_numpy()
                ax.scatter(x[success],values[success],marker=MARKERS[j],color=COLORS[method],alpha=.4,s=14)
                ax.scatter(x[~success],values[~success],marker='x',color=COLORS[method],alpha=.7,s=18)
                ax.scatter(i+(j-1)*.075,values.mean(),marker=MARKERS[j],color=COLORS[method],edgecolor='black',s=42,alpha=.65,zorder=5)
            ax.text(i,1.015,f'{int(data.terminated.sum())}/{len(data)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.axhline(expert[track],color='#333333',linestyle='--',linewidth=1.1)
        split='train' if track in reference['train_tracks'] else 'unseen'
        ax.set_title(f'{LABELS[track]}\n{split}',pad=24)
        ax.set_xticks(range(3),['Behavioral\nCloning','DAgger','RLPD'],fontsize=8)
        ax.set_ylabel('Episode steps');ax.margins(y=.13)
    box_handles=handles+[Line2D([],[],color='#333333',linestyle='--',label='Expert mean'),Line2D([],[],color='gray',marker='x',linestyle='',label='Timeout')]
    fig.legend(handles=box_handles,loc='lower center',ncol=8,frameon=False,fontsize=9)
    fig.suptitle('Episode steps - 3 training seeds × 20 test seeds; outlined markers show seed means')
    fig.tight_layout(rect=(0,.10,1,.94));fig.savefig(output/'multimap_step_distribution.svg',bbox_inches='tight');plt.close(fig)
    # Finishing speed is conditional on completion; attach completion counts to exported data.
    fig,axes=plt.subplots(1,2,figsize=(14,4.7),gridspec_kw={'width_ratios':[5,3]})
    for ax,split in zip(axes,('train','unseen')):
        current=reference['train_tracks'] if split=='train' else reference['unseen_tracks']
        for m,method in enumerate(METHODS):
            for i,track in enumerate(current):
                group=final[(final.method==method)&(final.track==track)].set_index('training_seed').reindex(seeds)
                values=group.relative_steps.to_numpy();x=i+(m-1)*.23
                if np.isfinite(values).all():ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),color=COLORS[method],fmt='_',capsize=4,markersize=13)
                for j,value in enumerate(values):
                    if np.isfinite(value):ax.scatter(x+(j-1)*.035,value,marker=MARKERS[j],color=COLORS[method],s=34)
                total=int(group.completed.sum());count=int(group.episodes.sum())
                ax.text(x,1.015,f'{total}/{count}',transform=ax.get_xaxis_transform(),ha='center',fontsize=7,rotation=60)
        ax.axhline(1,color='#333333',linestyle='--',linewidth=1.2)
        ax.set_xticks(range(len(current)),[LABELS[t] for t in current],rotation=20,ha='right')
        ax.set(title='Training maps' if split=='train' else 'Unseen maps',ylabel='Finishing steps / expert mean steps')
        ax.margins(y=.15)
        ax.set_title('Training maps' if split=='train' else 'Unseen maps',pad=38)
    relative_handles=handles+[Line2D([],[],color='#333333',linestyle='--',label='Expert mean (ratio 1)')]
    fig.legend(handles=relative_handles,loc='lower center',ncol=7,frameon=False,fontsize=9)
    fig.suptitle('Relative finishing steps - completed laps only; labels show completion counts',y=1.02)
    fig.tight_layout(rect=(0,.10,1,.94));fig.savefig(output/'multimap_relative_steps.svg',bbox_inches='tight');plt.close(fig)
    # Macro map averages per training seed, then mean/std over independently trained models.
    records=[]
    for (method,seed,split),d in final.groupby(['method','training_seed','split']):
        records.append(dict(method=method,training_seed=seed,split=split,completion_rate=d.completion_rate.mean(),
            relative_steps=d.relative_steps.mean() if d.relative_steps.notna().all() else np.nan,
            completed=int(d.completed.sum()),episodes=int(d.episodes.sum())))
    macro=pd.DataFrame(records);macro.to_csv(output/'multimap_macro_by_seed.csv',index=False)
    macro.groupby(['method','split']).agg(completion_mean=('completion_rate','mean'),completion_std=('completion_rate','std'),
        completed=('completed','sum'),episodes=('episodes','sum')).reset_index().to_csv(output/'multimap_macro_summary.csv',index=False)


def main(args):
    runs,reference=read_runs(args.root,args.training_seeds)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid');plt.rcParams.update({'svg.fonttype':'none','font.size':10})
    history=aligned_history(runs)
    history.to_csv(args.output_dir/'multimap_evaluation_history.csv',index=False)
    plot_evolution(history,reference,args.output_dir,args.training_seeds)
    if args.history_only:return
    jobs=make_jobs(args.root,runs,reference,args.training_seeds)
    if not args.plot_only:evaluate_all(jobs,args.workers)
    episodes,summary=final_tables(jobs,reference)
    assert len(episodes)==(len(METHODS)*len(args.training_seeds)+1)*len(reference['train_tracks']+reference['unseen_tracks'])*20
    plot_final(episodes,summary,reference,args.output_dir,args.training_seeds)
    (args.output_dir/'multimap_methodology.json').write_text(json.dumps(dict(source=str(args.root.resolve()),training_seeds=args.training_seeds,
        test_seeds=list(range(20000,20020)),selection='Validation-selected best per training seed; BC iteration 1; unseen excluded',
        alignment='DAgger: (iteration-1)*5000; RLPD: floor(actual steps/5000)*5000; missing checkpoints left missing',
        variability='Sample std across the three training-seed summaries, not a confidence interval; boxes pool 60 laps with seed markers',
        map_weighting='Equal map weighting within each split',timeouts='Included in step boxes; excluded from conditional finishing-speed means',
        relative_missing='No aggregate errorbar if any training seed has zero completions; individual available points retained',
        evaluation='CPU, two isolated simulator workers; original observation normalization restored'),indent=2))
    print('Saved multimap plots and tables:',args.output_dir,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'datasets/multimap/full5maps50k')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'datasets/plots')
    parser.add_argument('--training-seeds',type=int,nargs='+',default=[1000,2000,3000])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--history-only',action='store_true')
    args=parser.parse_args()
    if len(args.training_seeds)!=3 or len(set(args.training_seeds))!=3:parser.error('Require three distinct training seeds')
    if args.workers<1:parser.error('workers must be positive')
    main(args)
