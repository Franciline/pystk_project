"""Run: PYTHONPATH=src python src/unit_tests/test_svg_exports.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from main.plot_results import dagger_plot_results as plot_results
from utils.plot import plt_plot


if __name__=='__main__':
    with TemporaryDirectory() as directory:
        root=Path(directory)
        pd.DataFrame([dict(iteration=1,mse=.1,bce=.2,loss=.3)]).to_csv(root/'losses.csv',index=False)
        result=plot_results(root)
        assert result==[root/'plots/losses.svg']
        plt_plot([0,1],[0,2],'x','y','Test',root/'legacy.pdf')
        assert (root/'legacy.svg').exists()
        assert not list(root.rglob('*.pdf')) and not list(root.rglob('*.png'))
    print('SVG-only exports passed, including legacy caller filenames')
