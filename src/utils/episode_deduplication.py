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

