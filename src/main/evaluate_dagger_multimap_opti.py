"""Select a full25k DAgger checkpoint, test matched seeds on every map, and plot results."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.dagger_evaluation_opti import initialize_worker, evaluate_map
from utils.dagger_evaluation import append_metrics
from utils.multitrack import evaluation_tracks

LABELS={'abyss':'Abyss','volcano_island':'Volcan Island','sandtrack':'Shifting Sands',
        'hacienda':'Hacienda','minigolf':'Minigolf','snowtuxpeak':'Snow Peak','xr591':'XR591','zengarden':'Zen Garden'}


def plot_results(root):
    output=root/'final_test20'
    manifest=json.loads((output/'config.json').read_text())
    data=pd.read_csv(output/'episodes.csv')
    summary=pd.read_csv(output/'summary.csv').set_index('track')
    tracks=manifest['tracks']; config=manifest['task_config']
    figures=root/'plots/final_test20';figures.mkdir(parents=True,exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid');plt.rcParams['svg.fonttype']='none'
    fig,axes=plt.subplots(1,len(tracks),figsize=(3*len(tracks),4.5),squeeze=False)
    for ax,track in zip(axes[0],tracks):
        rows=data[data.track==track].sort_values('seed')
        finished=rows.terminated.astype(str).str.lower().eq('true').to_numpy()
        box=ax.boxplot([rows.n_steps],positions=[0],widths=.32,patch_artist=True,
            showfliers=False,medianprops=dict(color='#315b8a',linewidth=1.8))
        box['boxes'][0].set(facecolor='#4C72B0',alpha=.4)
        offsets=np.linspace(-.06,.06,len(rows))
        for mask,marker in ((finished,'o'),(~finished,'x')):
            ax.scatter(offsets[mask],rows.n_steps.to_numpy()[mask],marker=marker,color='#4C72B0',s=23,alpha=.8)
        split='train' if track in config['train_tracks'] else 'unseen'
        ax.set_title(f'{LABELS.get(track,track)}\n{split}',pad=28)
        ax.text(0,1.02,f'{finished.sum()}/{len(rows)} finish',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
        ax.set_xticks([0],['DAgger']);ax.set_xlim(-.4,.4);ax.set_ylabel('Episode steps')
        ax.margins(y=.15)
    fig.suptitle('Best five-map DAgger - 20 evaluation seeds')
    fig.legend(handles=[Line2D([],[],color='#4C72B0',marker='o',linestyle='',label='Completed lap'),
        Line2D([],[],color='#4C72B0',marker='x',linestyle='',label='Incomplete lap (timeout)')],
        loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(figures/'steps_boxplots.svg',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4))
    values=summary.reindex(tracks).completion_rate.to_numpy()*100
    bars=ax.bar(range(len(tracks)),values,color=['#4C72B0' if t in config['train_tracks'] else '#DD8452' for t in tracks],width=.55)
    for i,(bar,track) in enumerate(zip(bars,tracks)):
        row=summary.loc[track]
        ax.text(i,bar.get_height()+2,f"{int(row.completed)}/{int(row.episodes)}",ha='center')
    ax.set_xticks(range(len(tracks)),[LABELS.get(t,t)+'\n'+('train' if t in config['train_tracks'] else 'unseen') for t in tracks])
    ax.set(ylabel='Completion rate (%)',ylim=(0,110),title='Best five-map DAgger - completion over 20 seeds')
    fig.tight_layout();fig.savefig(figures/'completion_rate.svg',bbox_inches='tight');plt.close(fig)
    history=pd.read_csv(Path(manifest['source_run'])/'evaluation_by_track.csv')
    fig,ax=plt.subplots(figsize=(12,4))
    for track in config['unseen_tracks']:
        rows=history[history.track==track].sort_values('online_steps')
        ax.plot(rows.online_steps,rows.completion_rate*100,marker='o',label=LABELS.get(track,track))
    ax.axvline(manifest['best_evaluation']['online_steps'],color='gray',linestyle='--',label='Selected checkpoint')
    ax.set(xlabel='Online environment steps',ylabel='Completion rate (%)',ylim=(-3,103),title='Unseen-map completion during training - 5 evaluation seeds')
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,-.38),ncol=4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,pos:f'{x/1000:g}k'))
    fig.tight_layout();fig.savefig(figures/'unseen_completion_evolution.svg',bbox_inches='tight');plt.close(fig)
    print('Plots:',figures,flush=True)


def run(root,workers):
    candidates=[]
    for directory in sorted(root.glob('dagger_seed*')):
        if (directory/'best_evaluation.json').exists() and (directory/'actor_best.pt').exists():
            evaluation=json.loads((directory/'best_evaluation.json').read_text())
            candidates.append((directory,evaluation))
    if not candidates:raise ValueError('No saved best DAgger checkpoint found')
    source,best=min(candidates,key=lambda item:(-item[1]['completion_rate'],
        item[1]['finish_steps_mean'] if item[1]['finish_steps_mean'] is not None else float('inf')))
    config=json.loads((source/'config.json').read_text())
    output=root/'final_test20'
    if output.exists():raise FileExistsError(f'{output} exists; use --plot-only or a fresh results directory')
    checkpoint=root/'best_actor_dagger.pt'
    shutil.copy2(source/'actor_best.pt',checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==hashlib.sha256((source/'actor_best.pt').read_bytes()).hexdigest()
    selection=dict(source_run=str(source.resolve()),checkpoint=str(checkpoint.resolve()),sha256=digest,
        best_evaluation=best,selection_rule='Training-map completion, then macro finishing steps; unseen excluded')
    (root/'best_actor_selection.json').write_text(json.dumps(selection,indent=2))
    config['evaluation']['seeds']=list(range(20000,20020))
    config['diagnostics']['save_evaluation_episodes']=False
    tracks=evaluation_tracks(config)
    output.mkdir()
    (output/'config.json').write_text(json.dumps(dict(**selection,task_config=config,tracks=tracks,seeds=config['evaluation']['seeds']),indent=2))
    jobs=[(str(checkpoint.resolve()),config,t,str(output/'episodes')) for t in tracks]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=initialize_worker) as pool:
        for track,rows,summary in pool.map(evaluate_map,jobs):
            split='train' if track in config['train_tracks'] else 'unseen'
            append_metrics(output/'episodes.csv',[dict(track=track,split=split,**r) for r in rows])
            append_metrics(output/'summary.csv',[dict(track=track,split=split,**summary)])
            print(f"{track} ({split}): {summary['completed']}/20 finished; finish steps={summary['finish_steps_mean']}",flush=True)
    rows=pd.read_csv(output/'episodes.csv')
    assert len(rows)==20*len(tracks)
    assert rows.groupby('track').seed.apply(lambda x:sorted(x)==list(range(20000,20020))).all()
    plot_results(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'datasets/multimap/full25k')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    if args.plot_only:plot_results(args.root)
    else:run(args.root,args.workers)
