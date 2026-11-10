"""Run: PYTHONPATH=src python src/unit_tests/test_svg_exports.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from main.plot_dagger import plot_results
from main.plot_track_segments import plot_segments
from utils.plot import plt_plot


if __name__=='__main__':
    with TemporaryDirectory() as directory:
        root=Path(directory)
        pd.DataFrame([dict(iteration=1,mse=.1,bce=.2,loss=.3)]).to_csv(root/'losses.csv',index=False)
        result=plot_results(root)
        assert result==[root/'plots/losses.svg']
        geometry=root/'geometry';geometry.mkdir()
        pd.DataFrame(dict(segment_id=[0,1],length_m=[5,6])).to_csv(geometry/'a_segments.csv',index=False)
        plot_segments(geometry,root/'segments.png') # A caller's old suffix must not recreate PNG.
        assert (root/'segments.svg').exists()
        plt_plot([0,1],[0,2],'x','y','Test',root/'legacy.pdf')
        assert (root/'legacy.svg').exists()
        assert not list(root.rglob('*.pdf')) and not list(root.rglob('*.png'))
    print('SVG-only exports passed, including legacy caller filenames')
