"""Run with PYTHONPATH=src python src/unit_tests/test_trajectory_plot.py."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from main.compare_trajectories import plot_trajectories, widen_positions, road_offsets, MODELS, FILES


def test_plot():
    with TemporaryDirectory() as directory:
        root=Path(directory)
        (root/'config.json').write_text(json.dumps(dict(track='test',seed=1)))
        np.savez(root/'track.npz', nodes=np.array([[[0,0,0],[10,0,0]],[[10,0,0],[10,0,10]]]), widths=[4,4])
        metrics=[]
        for name, filename in zip(MODELS,FILES):
            pd.DataFrame(dict(x=[0,5,10],y=[0,0,0],z=[0,0,1])).to_csv(root/f'{filename}.csv',index=False)
            metrics.append(dict(model=name,n_steps=2,terminated=True))
        pd.DataFrame(metrics).to_csv(root/'summary.csv',index=False)
        plot_trajectories(root,root/'plots')
        assert (root/'plots/trajectories_test_seed1.pdf').stat().st_size>0
        assert (root/'plots/trajectories_test_seed1.svg').stat().st_size>0


if __name__ == '__main__':
    nodes = np.array([[[0.,0.,0.],[10.,0.,0.]]])
    assert np.allclose(widen_positions([[5,0,2]], nodes, 2), [[5,4]])
    assert np.allclose(widen_positions([[5,0,2]], nodes, 1), [[5,2]])
    bend = np.array([[[0.,0.,0.],[10.,0.,0.]], [[10.,0.,0.],[10.,0.,10.]]])
    offsets = road_offsets(bend, np.array([4.,4.]), 1.5)
    assert np.allclose(offsets[0,1], offsets[1,0])
    assert np.isfinite(offsets).all()
    test_plot()
    print('Trajectory plot checks passed')
