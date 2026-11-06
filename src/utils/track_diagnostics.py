"""World-space segment lengths; branch sums need not equal official track length."""
import csv
import json
from pathlib import Path
import numpy as np


def save_track_diagnostics(env, directory):
    directory = Path(directory) / 'track_geometry'
    directory.mkdir(parents=True, exist_ok=True)
    name = env.current_track
    lengths = np.linalg.norm(np.asarray(env.track.path_nodes)[:, 1] -
                             np.asarray(env.track.path_nodes)[:, 0], axis=1)
    with (directory / f'{name}_segments.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['segment_id', 'length_m'])
        writer.writerows(enumerate(lengths.tolist()))
    with (directory / f'{name}.json').open('w') as file:
        json.dump(dict(track=name, track_length_m=float(env.track.length),
                       segments=len(lengths), segment_length_mean_m=float(lengths.mean()),
                       segment_length_min_m=float(lengths.min()),
                       segment_length_max_m=float(lengths.max())), file, indent=2)
