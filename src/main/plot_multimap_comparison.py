"""Latest per-map evaluation: completion, expert-relative steps, rescues and stuck time."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def plot(dagger,rlpd,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tables=[]
    baseline=pd.read_csv(Path(dagger)/'expert_benchmark/summary.csv').set_index('track')['finish_steps_mean']
    for name,directory,step in [('DAgger',Path(dagger),'online_steps'),('RLPD',Path(rlpd),'env_steps')]:
        data=pd.read_csv(directory/'evaluation_by_track.csv')
        data=data[data[step]==data[step].max()].copy()
        data['method']=name
        data['expert_steps']=data['track'].map(baseline)
        data['relative_steps_mean']=data['finish_steps_mean']/data['expert_steps']
        data['relative_steps_std']=data['finish_steps_std']/data['expert_steps']
        tables.append(data)
    data=pd.concat(tables,ignore_index=True)
    data.to_csv(output/'comparison.csv',index=False)
    tracks=list(dict.fromkeys(data['track']))
    labels={'abyss':'Abyss','volcano_island':'Volcan Island','sandtrack':'Shifting Sands',
            'snowtuxpeak':'Snow Peak','xr591':'XR591','zengarden':'Zen Garden'}
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
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
        for suffix in ('pdf','svg'):
            fig.savefig(output/f'{field}.{suffix}',bbox_inches='tight')
        plt.close(fig)
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dagger',type=Path,required=True)
    parser.add_argument('--rlpd',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('datasets/plots/multimap_smoke'))
    a=parser.parse_args();plot(a.dagger,a.rlpd,a.output_dir)
