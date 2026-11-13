"""Check scheduled-step alignment, missing checkpoints and equal-map aggregation."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.multimap_reports import aligned_history,macro_history,plot_final,METHODS


def check():
    config=dict(train_tracks=['a'],unseen_tracks=['b'])
    with TemporaryDirectory(prefix='multimap-alignment-') as temporary:
        root=Path(temporary);runs={}
        for method in ('DAgger','RLPD'):
            directory=root/method;directory.mkdir()
            rows=[]
            # Intentionally omit the nominal 10k evaluation.
            for evaluation,actual in ((0,0),(1,5420),(3,15958)) if method=='DAgger' else ((1,5003),(3,15958)):
                for track in ('a','b'):
                    row=dict(track=track,split='train' if track=='a' else 'unseen',completion_rate=.8,
                             finish_steps_mean=600,finish_steps_std=10)
                    if method=='DAgger':row.update(iteration=evaluation+1,online_steps=actual)
                    else:row['env_steps']=actual
                    rows.append(row)
            pd.DataFrame(rows).to_csv(directory/'evaluation_by_track.csv',index=False)
            runs[method,1000]=(directory,config)
        history=aligned_history(runs,total=15000)
        assert history[history.target_online_steps==10000].completion_rate.isna().all()
        assert history[history.target_online_steps==15000].completion_rate.eq(.8).all()
        assert set(history[(history.method=='DAgger')&history.actual_online_steps.eq(5420)].target_online_steps)=={5000}
        assert not ((history.method=='RLPD')&history.target_online_steps.eq(0)).any()
        macro=macro_history(history,config)
        assert macro[macro.target_online_steps==10000].completion_rate.isna().all()
        assert macro[macro.target_online_steps==15000].completion_rate.eq(.8).all()
        # A missing map within a checkpoint must also make its split macro unavailable.
        two=dict(train_tracks=['a','b'],unseen_tracks=[])
        missing=history[(history.method=='DAgger')&history.target_online_steps.eq(5000)].copy()
        missing['split']='train'
        missing.loc[missing.track=='b','completion_rate']=np.nan
        assert macro_history(missing,two).completion_rate.isna().all()
    print('PASS: nominal step alignment, missing evaluations/maps, distinct DAgger zero point, macro safeguards')
    with TemporaryDirectory(prefix='multimap-plots-') as temporary:
        output=Path(temporary)
        reference=dict(train_tracks=['abyss'],unseen_tracks=['snowtuxpeak'])
        seeds=[1000,2000,3000];details=[];summaries=[]
        for track in ('abyss','snowtuxpeak'):
            for method in (*METHODS,'Expert (team4)'):
                for index,seed in enumerate(seeds if method!='Expert (team4)' else [None]):
                    success=20 if method=='Expert (team4)' else [0,10,20][index]
                    tags=dict(track=track,method=method,training_seed=seed,split='train' if track=='abyss' else 'unseen')
                    for i in range(20):
                        details.append(dict(**tags,seed=20000+i,n_steps=600+i if i<success else 1000,terminated=i<success))
                    summaries.append(dict(**tags,episodes=20,completed=success,completion_rate=success/20,
                                          finish_steps_mean=610 if success else np.nan))
        plot_final(pd.DataFrame(details),pd.DataFrame(summaries),reference,output,seeds)
        import xml.etree.ElementTree as ET
        svgs=list(output.glob('*.svg'))
        assert len(svgs)==3
        for svg in svgs:ET.parse(svg)
        assert not list(output.glob('*.png')) and not list(output.glob('*.pdf'))
        summary=pd.read_csv(output/'multimap_final_summary_by_seed.csv')
        assert summary.loc[summary.training_seed==1000,'relative_steps'].isna().all()
        assert (output/'multimap_expert_summary.csv').exists()
        print('PASS: final plots with zero/partial/full completion, missing finishing speeds, SVG-only export')


if __name__=='__main__':check()
