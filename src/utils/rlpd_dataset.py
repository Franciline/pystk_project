"""load the offline RLPD dataset written by src/main/collect_rlpd.py and build sub-datasets."""
from collections import Counter, defaultdict
import numpy as np
import gzip
from pathlib import Path
import pickle
from agents.rlpd_agent import action_to_vector
from utils.episode_deduplication import drop_duplicates

DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "rlpd"


def load_episodes(root=DATASET_DIR, agents=None, epsilons=None, tracks=None):
    """load episodes and their metadata matching the filter; None keeps all"""
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
    for TD targets use done = terminated; truncated only marks the time limit
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

def build_offline_buffer(episodes, encoder, normalizer, learned):
    """encode episodes into float32 arrays: obs, action, reward, next_obs, terminated.
    uses executed actions (they produced reward/next_obs), not expert labels.
    """
    obs, actions, rewards, next_obs, terminated = [], [], [], [], []
    for episode in episodes:
        #   encode each of the N+1 observations once, then reuse for obs and next_obs
        encoded = [normalizer.transform(encoder.encode(o)) for o in episode["observations"]]
        obs += encoded[:-1]
        next_obs += encoded[1:]
        actions += [action_to_vector(a, learned) for a in episode["actions"]]
        rewards += episode["rewards"]
        terminated += episode["terminated"]
    return {
        "obs": np.stack(obs),
        "action": np.stack(actions),
        "reward": np.array(rewards, dtype=np.float32),
        "next_obs": np.stack(next_obs),
        "terminated": np.array(terminated, dtype=np.float32),  # 1.0 = lap finished (done)
    }