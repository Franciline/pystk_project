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
            config['max_steps'], to_action,
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
