"""Test a single-map iteration-1 BC checkpoint and compare with saved DAgger/RLPD results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.dagger_actor import ActorPolicy
from main.DAgger import to_action
from main.plot_final_comparison import load_results
from pystk2_gymnasium.envs import STKRaceEnv,AgentSpec
from utils.dagger_evaluation import evaluate_policy,append_metrics

ROOT=Path(__file__).resolve().parents[2]


def evaluate(args):
    output=args.bc_results
    if output.exists():raise FileExistsError(f'{output} already exists; use --plot-only to reuse saved results')
    policy=ActorPolicy.load(args.checkpoint,device=args.device)
    seeds=list(range(20000,20020))
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint.resolve()),
        sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),role='Behavioral cloning: DAgger iteration 1',
        track='abyss',step_size=.1,max_steps=1000,seeds=seeds,device=str(policy.device)),indent=2))
    env=STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track='abyss',num_kart=1,laps=1,step_size=.1)
    try:
        rows,summary=evaluate_policy(env,policy,seeds,1000,to_action)
        append_metrics(output/'final_test_episodes.csv',[dict(checkpoint='bc',**row) for row in rows])
        append_metrics(output/'final_test.csv',[dict(checkpoint='bc',**summary)])
        print('BC:',summary,flush=True)
    finally:env.close()


def plot(args):
    expert=load_results(args.expert,checkpoint='team4')
    seeds=expert.index
    if len(seeds)!=20 or not expert.terminated.all():raise ValueError('Require 20 completed expert tests')
    training_seeds=getattr(args,'training_seeds',None)
    if training_seeds:
        if len(set(training_seeds))!=len(training_seeds):raise ValueError('Duplicate training seeds')
        data={}
        for name,prefix,checkpoint in [('BC','bc_gaussian','bc'),('DAgger','dagger_offline_gaussian','best'),('RLPD','rlpd','best')]:
            frames=[]
            for seed in training_seeds:
                rows=load_results(args.protocol_dir/f'{prefix}_seed{seed}',checkpoint=checkpoint)
                if not rows.index.equals(seeds):raise ValueError(f'Mismatched race seeds: {name}, training seed {seed}')
                frames.append(rows)
            data[name]=pd.concat(frames,keys=training_seeds,names=['training_seed','seed'])
    else:
        data={'BC':load_results(args.bc_results,checkpoint='bc'),
              'DAgger':load_results(args.dagger),'RLPD':load_results(args.rlpd)}
        if not all(rows.index.equals(seeds) for rows in data.values()):raise ValueError('Test seeds must match')
    plt.style.use('seaborn-v0_8-darkgrid');plt.rcParams.update({'svg.fonttype':'none','font.size':11,'axes.titlesize':15,'axes.labelsize':12})
    fig,ax=plt.subplots(figsize=(6,5.5))
    distributions=[rows.n_steps.to_numpy(dtype=float) for rows in data.values()]
    box=ax.boxplot(distributions,tick_labels=['Behavioral Cloning' if name == 'BC' else name for name in data],patch_artist=True,showfliers=False,widths=.28,
        medianprops=dict(color='#222222',linewidth=2),whiskerprops=dict(color='#333333',linewidth=1.5),
        capprops=dict(color='#333333',linewidth=1.5))
    colors=['#55A868','#4C72B0','#DD8452'];summaries=[]
    for index,((name,rows),steps,color,patch) in enumerate(zip(data.items(),distributions,colors,box['boxes']),start=1):
        patch.set(facecolor=color,edgecolor=color,alpha=.6)
        finished=rows.terminated.to_numpy()
        if training_seeds:
            from matplotlib.lines import Line2D
            markers=['o','s','^','D','v','P']
            if len(training_seeds)>len(markers):raise ValueError('At most six training seeds supported')
            for group,(seed,center) in enumerate(zip(training_seeds,np.linspace(-.10,.10,len(training_seeds)))):
                mask=rows.index.get_level_values('training_seed')==seed
                values=steps[mask]
                success=finished[mask]
                offsets=center+np.linspace(-.025,.025,len(values))
                ax.scatter(index+offsets[success],values[success],marker=markers[group],color=color,
                    edgecolor='white',linewidth=.4,alpha=.55,s=20,zorder=3)
                mean_color=tuple(np.asarray(matplotlib.colors.to_rgb(color))*.55)
                ax.scatter(index+center,values.mean(),marker=markers[group],color=mean_color,
                    edgecolor='black',linewidth=.8,s=60,alpha=.6,zorder=5)
                if (~success).any():
                    ax.scatter(index+offsets[~success],values[~success],marker='x',color=color,s=28,zorder=3)
        else:
            offsets=np.linspace(-.10,.10,len(seeds))
            for mask,marker in ((finished,'o'),(~finished,'x')):
                ax.scatter(index+offsets[mask],steps[mask],marker=marker,color=color,
                    edgecolor='white' if marker=='o' else None,linewidth=.5 if marker=='o' else 1.2,alpha=.9,s=34,zorder=3)
        finish=steps[finished]
        summaries.append(dict(method=name,episodes=len(steps),completed=int(finished.sum()),completion_rate=float(finished.mean()),
            steps_mean=steps.mean(),steps_std=steps.std(ddof=1),finish_steps_mean=finish.mean() if len(finish) else None,
            finish_steps_std=finish.std(ddof=1) if len(finish)>1 else None))
    expert_mean=float(expert.n_steps.mean())
    ax.axhline(expert_mean,color='#333333',linestyle='--',linewidth=1.4,label=f'Expert (team4) mean: {expert_mean:.2f} steps')
    ax.legend(frameon=False,fontsize=10)
    title=(f'Distribution of episode steps\n{len(training_seeds)} training seeds × {len(seeds)} test seeds' if training_seeds
           else f'Distribution of episode steps across {len(seeds)} test seeds')
    ax.set(title=title,ylabel='Episode steps')
    if training_seeds:
        handles=[Line2D([],[],color='gray',marker=markers[i],linestyle='',label=str(seed)) for i,seed in enumerate(training_seeds)]
        fig.legend(handles=handles,title='Training seed (large outlined markers = means)',loc='lower center',ncol=len(training_seeds),frameon=False,fontsize=9,title_fontsize=9)
        fig.tight_layout(rect=(0,.11,1,1))
    else:
        fig.tight_layout()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/'bc_dagger_rlpd_step_distribution.svg'
    fig.savefig(target,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(summaries).to_csv(args.output_dir/'bc_dagger_rlpd_final_summary.csv',index=False)
    paired=pd.DataFrame({name.lower()+'_steps':rows.n_steps for name,rows in data.items()})
    paired.reset_index().to_csv(args.output_dir/'bc_dagger_rlpd_final_paired.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False));print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'datasets/protocol/dagger_offline_gaussian_seed1000/actor_iteration_1.pt')
    parser.add_argument('--bc-results',type=Path,default=ROOT/'datasets/protocol/bc_gaussian_seed1000')
    parser.add_argument('--dagger',type=Path,default=ROOT/'datasets/protocol/best_dagger_gaussian')
    parser.add_argument('--rlpd',type=Path,default=ROOT/'datasets/protocol/rlpd_seed1000')
    parser.add_argument('--expert',type=Path,default=ROOT/'datasets/protocol/expert_test')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'datasets/plots')
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--training-seeds',type=int,nargs='+',help='Pool saved tests across training seeds (plot-only)')
    parser.add_argument('--protocol-dir',type=Path,default=ROOT/'datasets/protocol')
    args=parser.parse_args()
    if args.training_seeds and not args.plot_only:parser.error('--training-seeds requires --plot-only')
    if not args.plot_only:evaluate(args)
    plot(args)
