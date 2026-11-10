"""Shared map scheduling and evaluation; old single-track configurations still work."""
from pathlib import Path
import numpy as np
from utils.dagger_evaluation import evaluate_policy


def train_tracks(config):
    return list(config.get('train_tracks') or [config['track']])


def evaluation_tracks(config):
    trained = train_tracks(config)
    unseen = list(config.get('unseen_tracks') or [])
    if set(trained) & set(unseen):
        raise ValueError('Training and unseen tracks must be disjoint')
    return trained + [t for t in unseen if t not in trained]


def next_track(tracks, steps):
    """Least-collected map first; finish each episode before switching maps."""
    return min(tracks,key=lambda track: steps.get(track,0))


def evaluate_tracks(env, policy, config, record_dir, to_action):
    rows, summaries = [], []
    trained = train_tracks(config)
    for track in evaluation_tracks(config):
        env.default_track = track
        detail, summary = evaluate_policy(env,policy,config['evaluation']['seeds'],
            track_step_limit(config,track), to_action,
            diagnostics=config.get('diagnostics'),
            record_dir=Path(record_dir)/track if config.get('diagnostics',{}).get('save_evaluation_episodes',True) else None)
        split = 'train' if track in trained else 'unseen'
        rows.extend(dict(track=track,split=split,**r) for r in detail)
        summaries.append(dict(track=track,split=split,**summary))
    selected = [s for s in summaries if s['split']=='train']
    if len(selected)==1:
        aggregate = {k:v for k,v in selected[0].items() if k not in ('track','split')}
    else:
        aggregate = dict(episodes=sum(s['episodes'] for s in selected),
            completed=sum(s['completed'] for s in selected),
            completion_rate=float(np.mean([s['completion_rate'] for s in selected])),
            timeouts=sum(s['timeouts'] for s in selected))
        for key in selected[0]:
            if key.endswith(('_mean','_std')):
                values=[s[key] for s in selected]
                aggregate[key]=float(np.mean(values)) if key.endswith('_mean') and all(v is not None for v in values) else None
        # Standard deviations remain per-map; do not mix different track lengths.
    return rows,summaries,aggregate


def track_step_limit(config, track):
    """A frozen map-specific timeout, or the original single-map fallback."""
    limit = int(config.get('max_steps_by_track',{}).get(track,config['max_steps']))
    if limit < 1:
        raise ValueError('Episode step limits must be positive')
    return limit


def prepare_config(config):
    """Resolve the shared expert calibration before any dataset/training is started."""
    if config.get('track_timeouts',{}).get('mode') != 'expert':
        return config
    from copy import deepcopy
    from main.benchmark_expert import prepare_protocol
    reference, limits = prepare_protocol(config)
    result=deepcopy(config)
    result['expert_reference_dir']=str(reference)
    result['max_steps_by_track']=limits
    return result


def save_expert_reference(config, output_dir):
    """Reuse the exact common baseline in each run; legacy benchmarks remain supported."""
    import shutil
    if config.get('expert_reference_dir'):
        source=Path(config['expert_reference_dir']).resolve()
        destination=(Path(output_dir)/'expert_benchmark').resolve()
        if source==destination or source in destination.parents:
            raise ValueError('Expert reference must be outside the run output directory')
        shutil.copytree(source,destination)
        return
    if config.get('expert_benchmark'):
        from main.benchmark_expert import run_benchmark
        benchmark=config['expert_benchmark']
        run_benchmark(config,Path(output_dir)/'expert_benchmark',
            list(range(benchmark['seed_start'],benchmark['seed_start']+benchmark['runs'])))


class MapCollection:
    """Online seed counters and cumulative exposure, independently for each map."""
    def __init__(self, tracks, seed_start):
        self.tracks=list(tracks)
        self.seed_start=seed_start
        self.episodes={track:0 for track in tracks}
        self.steps={track:0 for track in tracks}

    def next_seed(self, track):
        return self.seed_start+self.episodes[track]

    def record(self, track, steps):
        self.episodes[track]+=1
        self.steps[track]+=steps

    def summary(self, interval_episodes, interval_steps, **checkpoint):
        total=sum(self.steps.values()); interval_total=sum(interval_steps.values())
        return [dict(**checkpoint,track=track,
            interval_trajectories=interval_episodes.get(track,0),
            interval_steps=interval_steps.get(track,0),
            interval_step_fraction=interval_steps.get(track,0)/interval_total if interval_total else 0.,
            cumulative_trajectories=self.episodes[track],cumulative_steps=self.steps[track],
            cumulative_step_fraction=self.steps[track]/total if total else 0.,
            next_episode_seed=self.next_seed(track)) for track in self.tracks]


def report_collection(path, rows):
    from utils.dagger_evaluation import append_metrics
    append_metrics(path,rows)
    for row in rows:
        print(f"  {row['track']}: {row['interval_trajectories']} trajectories / "
              f"{row['interval_steps']} steps this interval; cumulative "
              f"{row['cumulative_trajectories']} / {row['cumulative_steps']} steps "
              f"({row['cumulative_step_fraction']:.1%})",flush=True)
