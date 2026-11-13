"""Single-map sample efficiency: separate completion and successful-lap step figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
COLORS={'Behavioral Cloning':'#55A868','DAgger':'#4C72B0','RLPD':'#DD8452'}


def load_history(protocol,seeds,interval=5000,total=50000):
    tables=[]
    for method,prefix in [('DAgger','dagger_offline_gaussian'),('RLPD','rlpd')]:
        for seed in seeds:
            directory=protocol/f'{prefix}_seed{seed}'
            data=pd.read_csv(directory/'evaluation.csv')
            data['actual_online_steps']=data.online_steps if method=='DAgger' else data.env_steps
            data['target_online_steps']=((data.iteration-1)*interval if method=='DAgger' else
                                         (data.env_steps//interval)*interval).astype(int)
            if data.target_online_steps.duplicated().any():raise ValueError(f'Duplicate evaluation targets at {directory}')
            expected=list(range(0 if method=='DAgger' else interval,total+1,interval))
            if set(data.target_online_steps)-set(expected):raise ValueError(f'Unexpected evaluation targets at {directory}')
            # Completion counts are authoritative: never treat a timeout's steps as finishing time.
            data.loc[data.completed.eq(0),'finish_steps_mean']=np.nan
            data=data.set_index('target_online_steps').reindex(expected).rename_axis('target_online_steps').reset_index()
            data['method']=method;data['training_seed']=seed
            tables.append(data)
    return pd.concat(tables,ignore_index=True)


def plot(protocol,output,seeds):
    history=load_history(protocol,seeds)
    output.mkdir(parents=True,exist_ok=True)
    history.to_csv(output/'singlemap_learning_history.csv',index=False)
    initial=history[(history.method=='DAgger')&history.target_online_steps.eq(0)]
    if len(initial)!=len(seeds):raise ValueError('Missing BC baseline')
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'svg.fonttype':'none','font.size':11})
    summaries=[]
    for metric,title,ylabel,filename,multiplier in [
        ('completion_rate','Single-map completion during training','Completion rate (%)','singlemap_completion_evolution.svg',100),
        ('finish_steps_mean','Single-map finishing steps - mean ± std across training seeds','Steps to finish','singlemap_finishing_steps_evolution.svg',1),
    ]:
        fig,ax=plt.subplots(figsize=(8.5,3.2) if metric=='finish_steps_mean' else (11,4.3))
        baseline=initial[metric]
        if baseline.notna().all():
            mean=baseline.mean()*multiplier;std=baseline.std(ddof=1)*multiplier
            ax.axhline(mean,color=COLORS['Behavioral Cloning'],linestyle='--',linewidth=1.7,label='Behavioral Cloning')
            low,high=mean-std,mean+std
            if metric=='completion_rate':low,high=max(0,low),min(100,high)
            ax.axhspan(low,high,color=COLORS['Behavioral Cloning'],alpha=.10)
        for method in ('DAgger','RLPD'):
            rows=history[history.method==method]
            grouped=rows.groupby('target_online_steps')[metric].agg(['mean','std','count']).reset_index()
            # No averaging over only the successful models when a training seed has no finishing lap.
            grouped.loc[grouped['count']!=len(seeds),['mean','std']]=np.nan
            x=grouped.target_online_steps.to_numpy();mean=grouped['mean'].to_numpy()*multiplier
            std=grouped['std'].to_numpy()*multiplier
            ax.plot(x,mean,color=COLORS[method],marker='o' if method=='DAgger' else '^',
                markersize=7 if method=='DAgger' else 4,markerfacecolor='none' if method=='DAgger' else COLORS[method],
                linestyle='-' if method=='DAgger' else '-.',linewidth=1.6,label=method)
            low,high=mean-std,mean+std
            if metric=='completion_rate':low=np.maximum(0,low);high=np.minimum(100,high)
            ax.fill_between(x,low,high,color=COLORS[method],alpha=.17)
            grouped['method']=method;grouped['metric']=metric;summaries.append(grouped)
        if metric=='finish_steps_mean':
            reference=pd.read_csv(protocol/f'dagger_offline_gaussian_seed{seeds[0]}'/'expert_benchmark/summary.csv')
            expert_mean=float(reference.loc[reference.track=='abyss','finish_steps_mean'].iloc[0])
            ax.axhline(expert_mean,color='#333333',linestyle=':',linewidth=1.6,
                       label=f'Expert (team4): {expert_mean:.1f} steps')
        if metric=='completion_rate':ax.set_ylim(-3,103)
        ax.set(title=title,xlabel='Target online steps',ylabel=ylabel,xlim=(0,50000))
        ax.set_xticks(np.arange(0,50001,5000));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:g}k'))
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4 if metric=='finish_steps_mean' else 3,frameon=False,
                   fontsize=9 if metric=='finish_steps_mean' else 11)
        fig.tight_layout(rect=(0,.11,1,1));fig.savefig(output/filename,bbox_inches='tight');plt.close(fig)
        print(output/filename)
    pd.concat(summaries,ignore_index=True).to_csv(output/'singlemap_learning_summary.csv',index=False)
    initial[['training_seed','episodes','completed','completion_rate','finish_steps_mean']].to_csv(output/'singlemap_bc_learning_baseline.csv',index=False)
    notes=('Source: single-map Abyss Gaussian DAgger and RLPD runs, training seeds '+str(seeds)+'.\n'
           'Every recorded evaluation is used, not a running best checkpoint.\n'
           'Targets: DAgger (iteration-1)*5000; RLPD floor(actual online steps/5000)*5000. Actual counts are preserved.\n'
           'Missing targets remain missing. RLPD has no invented zero-step value.\n'
           'BC is the initial offline-only DAgger validation baseline, held constant.\n'
           'Means and sample standard deviations are across training-seed evaluation means, not individual laps or confidence intervals.\n'
           'Finishing steps use successful laps only. A combined point is omitted if any training seed has no completed lap.\n')
    (output/'singlemap_learning_methodology.txt').write_text(notes)
    assert history.groupby(['method','training_seed']).episodes.first().eq(5).all()
    return history


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-dir',type=Path,default=ROOT/'datasets/protocol')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'datasets/plots')
    parser.add_argument('--training-seeds',type=int,nargs='+',default=[1000,2000,3000])
    args=parser.parse_args()
    if len(args.training_seeds)<2 or len(set(args.training_seeds))!=len(args.training_seeds):parser.error('Require distinct training seeds, at least two')
    plot(args.protocol_dir,args.output_dir,args.training_seeds)
