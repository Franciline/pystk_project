"""Shared offline episode deduplication for RLPD and DAgger."""
import pickle
import numpy as np


def drop_duplicates(episodes):
    """keep the first of each group of identical episodes (same track, actions, rewards, progress)"""
    seen, unique = set(), []
    for episode in episodes:
        # progress along the track at each step: catches same actions but different outcome
        progress = [float(np.ravel(obs["distance_down_track"])[0]) for obs in episode["observations"]]
        key = pickle.dumps((episode.get("metadata",{}).get("track"),episode["actions"],episode["rewards"],progress))
        if key not in seen:  # first time we see this exact episode
            seen.add(key)
            unique.append(episode)
    return unique



def deduplication_report(episodes, unique):
    """Per-map/epsilon counts and transition totals, before and after exact dedup."""
    groups=sorted({(e['metadata']['track'],float(e['metadata']['epsilon'])) for e in episodes})
    rows=[]
    for track,epsilon in groups:
        before=[e for e in episodes if e['metadata']['track']==track and float(e['metadata']['epsilon'])==epsilon]
        after=[e for e in unique if e['metadata']['track']==track and float(e['metadata']['epsilon'])==epsilon]
        rows.append(dict(track=track,epsilon=epsilon,trajectories_before=len(before),
            trajectories_after=len(after),steps_before=sum(len(e['actions']) for e in before),
            steps_after=sum(len(e['actions']) for e in after)))
    return rows
