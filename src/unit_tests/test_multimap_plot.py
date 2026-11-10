"""Run with PYTHONPATH=src python src/unit_tests/test_multimap_plot.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from main.plot_multimap_comparison import plot


if __name__=='__main__':
    with TemporaryDirectory() as directory:
        root=Path(directory)
        for method,step in [('dagger','online_steps'),('rlpd','env_steps')]:
            folder=root/method;(folder/'expert_benchmark').mkdir(parents=True)
            rows=[]
            for track,split in [('abyss','train'),('xr591','unseen')]:
                rows.append(dict(track=track,split=split,**{step:5000},
                    finish_steps_mean=110,finish_steps_std=5,completion_rate=1,
                    rescue_count_mean=0,rescue_count_std=0,stuck_duration_s_mean=0,stuck_duration_s_std=0))
            pd.DataFrame(rows).to_csv(folder/'evaluation_by_track.csv',index=False)
            pd.DataFrame([dict(track=t,finish_steps_mean=100) for t in ['abyss','xr591']]).to_csv(folder/'expert_benchmark/summary.csv',index=False)
        plot(root/'dagger',root/'rlpd',root/'plots')
        assert len(list((root/'plots').glob('*.pdf')))==4
        assert len(list((root/'plots').glob('*.svg')))==4
        assert (pd.read_csv(root/'plots/comparison.csv').relative_steps_mean==1.1).all()
        for method in ['dagger','rlpd']:
            p=root/method/'evaluation_by_track.csv';data=pd.read_csv(p)
            data[['finish_steps_mean','finish_steps_std','rescue_count_mean','rescue_count_std']]=float('nan')
            data['completion_rate']=0
            data.to_csv(p,index=False)
        plot(root/'dagger',root/'rlpd',root/'failed_plots')
    print('Multi-map plot checks passed')
