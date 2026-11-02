"""load the offline RLPD dataset written by src/main/collect_rlpd.py and build sub-datasets."""
from collections import Counter, defaultdict
import numpy as np
import gzip
from pathlib import Path
import pickle

DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "rlpd"


def load_episodes(root=DATASET_DIR, agents=None, epsilons=None, tracks=None):
    """load episodes and their metadata that matches every given filter; None keeps all.

    files are pickles: load only self-collected, trusted data.
    """
    episodes = []
    for path in sorted(Path(root).rglob("*.pkl.gz")):
        with gzip.open(path, "rb") as file:
            episode = pickle.load(file)
        meta = episode["metadata"]
        if ((agents is None or meta["agent"] in agents)
                and (epsilons is None or meta["epsilon"] in epsilons)
                and (tracks is None or meta["track"] in tracks)):
            episodes.append(episode)
    return episodes


def episode_id(meta):
    return f"{meta['track']}/{meta['agent']}_eps{meta['epsilon']}_seed{meta['seed']}"


def to_transitions(episodes):
    """flatten episodes into columns: one list per field, one entry per transition.

    for TD targets use done = terminated; truncated only marks the time limit.
    """
    columns = {key: [] for key in (
        "obs", "action", "expert_action", "reward", "next_obs",
        "terminated", "truncated", "episode_id", "step_id",
    )}
    for episode in episodes:
        n = len(episode["actions"])
        columns["obs"] += episode["observations"][:-1]
        columns["next_obs"] += episode["observations"][1:]
        columns["action"] += episode["actions"]
        columns["expert_action"] += episode["expert_actions"]
        columns["reward"] += episode["rewards"]
        columns["terminated"] += episode["terminated"]
        columns["truncated"] += episode["truncated"]
        columns["episode_id"] += [episode_id(episode["metadata"])] * n
        columns["step_id"] += list(range(n))
    return columns