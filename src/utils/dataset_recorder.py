"""Record one kart's episodes, preserving raw observation/action dictionaries."""

from copy import deepcopy
import gzip
from pathlib import Path
import pickle


class DatasetRecorder:
    def __init__(self):
        self.episode = None

    def start_episode(self, observation, metadata=None):
        """Start after reset(); do not discard an unsaved episode."""
        if self.episode is not None:
            raise RuntimeError("Save the current episode before starting another")
        self.episode = {
            "observations": [deepcopy(observation)],
            "actions": [],
            "expert_actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "metadata": deepcopy(metadata) if metadata is not None else {},
        }

    def record_step(self, action, reward, next_observation, terminated, truncated,
                    expert_action=None):
        """Append immediately after step(); omitted expert action means expert drove.

        Copy action/label before step() if the environment mutates action inputs.
        Mark a runner-imposed step limit as truncated on its final transition.
        """
        if self.episode is None:
            raise RuntimeError("Call start_episode() first")
        episode = self.episode
        if episode["terminated"] and (
            episode["terminated"][-1] or episode["truncated"][-1]
        ):
            raise RuntimeError("Episode has already ended")
        values = {
            "observations": deepcopy(next_observation),
            "actions": deepcopy(action),
            "expert_actions": deepcopy(action if expert_action is None else expert_action),
            "rewards": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        for key, value in values.items():
            episode[key].append(value)

    def save_episode(self, path):
        """Save a completed episode as gzip-compressed pickle; refuse overwrites.

        Load trusted files only, using pickle.load(gzip.open(path, 'rb')).
        """
        if self.episode is None:
            raise RuntimeError("No episode to save")
        if not self.episode["terminated"] or not (
            self.episode["terminated"][-1] or self.episode["truncated"][-1]
        ):
            raise RuntimeError("Episode must finish or be marked truncated before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
            pickle.dump(self.episode, compressed, protocol=pickle.HIGHEST_PROTOCOL)
        self.episode = None
        return path
