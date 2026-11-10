"""Latest per-map evaluation: completion, expert-relative steps, rescues and stuck time."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


def plot(dagger,rlpd,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tables=[];histories=[]
    baseline=pd.read_csv(Path(dagger)/'expert_benchmark/summary.csv').set_index('track')['finish_steps_mean']
    for name,directory,step in [('DAgger',Path(dagger),'online_steps'),('RLPD',Path(rlpd),'env_steps')]:
        data=pd.read_csv(directory/'evaluation_by_track.csv')
        data['method']=name
        data['online_steps']=data[step]
        data['expert_steps']=data['track'].map(baseline)
        data['relative_steps_mean']=data['finish_steps_mean']/data['expert_steps']
        data['relative_steps_std']=data['finish_steps_std']/data['expert_steps']
        histories.append(data.copy())
        tables.append(data[data['online_steps']==data['online_steps'].max()].copy())
    data=pd.concat(tables,ignore_index=True)
    data.to_csv(output/'comparison.csv',index=False)
    tracks=list(dict.fromkeys(data['track']))
    labels={'abyss':'Abyss','volcano_island':'Volcan Island','sandtrack':'Shifting Sands',
            'snowtuxpeak':'Snow Peak','xr591':'XR591','zengarden':'Zen Garden'}
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'svg.fonttype':'none'})
    for field,title,ylabel in [
        ('relative_steps','Relative finishing steps','Actor steps / expert mean steps'),
        ('completion_rate','Lap completion','Completion rate'),
        ('rescue_count','Rescues per lap','Rescue count'),
        ('stuck_duration_s','Time stuck per lap','Stuck duration (s)'),
    ]:
        fig,ax=plt.subplots(figsize=(13,4.3))
        handles=[Patch(facecolor=color,label=name) for name,color in [('DAgger','#4C72B0'),('RLPD','#DD8452')]]
        upper=1.
        for index,(name,color) in enumerate([('DAgger','#4C72B0'),('RLPD','#DD8452')]):
            rows=data[data.method==name].set_index('track').reindex(tracks)
            column=field if field=='completion_rate' else field+'_mean'
            values=rows[column].to_numpy(dtype=float)
            errors=np.zeros(len(rows)) if field=='completion_rate' else rows[field+'_std'].fillna(0).to_numpy()
            x=np.arange(len(tracks))+(index-.5)*.32
            finite=np.isfinite(values)
            if finite.any(): upper=max(upper,float(np.max(values[finite]+errors[finite])))
            ax.bar(x[finite],values[finite],width=.3,yerr=errors[finite],capsize=3,color=color,label=name)
            for position in x[~finite]:
                ax.text(position,0,'no ratio' if field=='relative_steps' else 'unavailable',
                        rotation=90,va='bottom',ha='center',fontsize=8)
        if field=='relative_steps':
            handles.append(ax.axhline(1,color='#333333',linestyle='--',linewidth=1,label='Expert (team4)'))
        ax.set_ylim(0,1.08 if field=='completion_rate' else upper*1.15)
        ax.set_xlim(-.6,len(tracks)-.4)
        splits=data.drop_duplicates('track').set_index('track')['split']
        ax.set_xticks(range(len(tracks)),[labels.get(t,t)+'\n'+splits[t] for t in tracks])
        ax.set(title=title,xlabel='Map',ylabel=ylabel)
        ax.legend(handles=handles,frameon=False);fig.tight_layout()
        for suffix in ('svg',):
            fig.savefig(output/f'{field}.{suffix}',bbox_inches='tight')
        plt.close(fig)
    history=pd.concat(histories,ignore_index=True)
    history.to_csv(output/'evaluation_history.csv',index=False)
    for track in tracks:
        for metric,title,ylabel in [('relative_steps','Relative finishing steps','Actor steps / expert mean steps'),
                                    ('completion_rate','Lap completion','Completion rate')]:
            fig,ax=plt.subplots(figsize=(11,4.3))
            for name,color in [('DAgger','#4C72B0'),('RLPD','#DD8452')]:
                rows=history[(history.method==name)&(history.track==track)].sort_values('online_steps')
                x=rows.online_steps.to_numpy(dtype=float)
                column='relative_steps_mean' if metric=='relative_steps' else metric
                y=rows[column].to_numpy(dtype=float)
                ax.plot(x,y,color=color,marker='o',label=name)
                if metric=='relative_steps':
                    std=rows.relative_steps_std.to_numpy(dtype=float)
                    ax.fill_between(x,y-std,y+std,color=color,alpha=.18)
            if metric=='relative_steps':ax.axhline(1,color='#333333',linestyle='--',label='Expert (team4)')
            else:ax.set_ylim(-.03,1.08)
            ax.set(title=f'{title} - {labels.get(track,track)}',xlabel='Online environment steps',ylabel=ylabel)
            ax.set_xlim(left=0)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos: f'{value/1000:g}k' if value>=1000 else f'{value:g}'))
            ax.legend(frameon=False);fig.tight_layout()
            for suffix in ('svg',):
                fig.savefig(output/f'evolution_{metric}_{track}.{suffix}',bbox_inches='tight')
            plt.close(fig)
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dagger',type=Path,required=True)
    parser.add_argument('--rlpd',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('datasets/plots/multimap_expert_limits'))
    a=parser.parse_args();plot(a.dagger,a.rlpd,a.output_dir)
