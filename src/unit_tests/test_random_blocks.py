"""Run: PYTHONPATH=src python src/unit_tests/test_random_blocks.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
import gzip
import pickle
from main.single_track_race_display import single_race


class Space:
    def seed(self, seed): self.count = 0
    def __getitem__(self, key): return self
    def sample(self):
        self.count += 1
        return dict(acceleration=self.count / 10, steer=0, brake=False,
                    drift=False, nitro=False, fire=False, rescue=True)


class Env:
    current_track = 'test'
    def __init__(self): self.action_space = Space(); self.actions = []
    def reset(self, seed): return {'0': {}}, {}
    def step(self, actions):
        self.actions.append(actions['0'])
        return {'0': {}}, 0, len(self.actions) == 6, False, {'reward': {'0': 0}}


class Expert:
    def __init__(self): self.calls = 0
    def choose_action(self, obs):
        self.calls += 1
        return dict(acceleration=1., steer=0, brake=False, drift=False,
                    nitro=False, fire=False, rescue=False)


def test_blocks():
    with TemporaryDirectory() as directory:
        env, expert = Env(), Expert()
        path = Path(directory) / 'episode.pkl.gz'
        single_race(env, [expert], ['expert'], path, seed=2, max_steps=10,
                    epsilon=1., hold_steps=3)
        assert [a['acceleration'] for a in env.actions] == [.1,.2,.3,.4,.5,.6]
        assert not any(a['rescue'] for a in env.actions)
        assert expert.calls == 6
        with gzip.open(path, 'rb') as f: episode = pickle.load(f)
        assert all(a['acceleration'] == 1 for a in episode['expert_actions'])
        env, expert = Env(), Expert()
        single_race(env, [expert], ['expert'], None, seed=2, epsilon=0)
        assert env.action_space.count == 0
        assert all(a['acceleration'] == 1 for a in env.actions)


if __name__ == '__main__':
    test_blocks()
    print('Random block checks passed')
