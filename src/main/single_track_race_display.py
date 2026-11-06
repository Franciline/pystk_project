"""Display a saved actor or Team4 for one lap, optionally recording."""

import argparse
import json
from copy import deepcopy
import sys, os
import time
from pathlib import Path

# Append the "src" folder to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from agents.team4.agent4 import Agent4
from pystk2_gymnasium.envs import STKRaceMultiEnv, AgentSpec
from pystk2_gymnasium.definitions import CameraMode
from utils.dataset_recorder import DatasetRecorder

MAX_TEAMS = 1
MAX_STEPS = 1000
NB_RACES = 1

# Make AgentSpec hashable.
def agent_spec_hash(self):
    return hash((self.name, self.rank_start, self.use_ai, self.camera_mode))
AgentSpec.__hash__ = agent_spec_hash

# Create agents specifications.
agents_specs = [
    AgentSpec(name=f"Team{4}", rank_start=0, use_ai=False, camera_mode=CameraMode.ON) # team$ only
]
# agents_specs = [
#     AgentSpec(name=f"Team{i+1}", rank_start=i, use_ai=False, camera_mode=CameraMode.ON) for i in range(MAX_TEAMS)
# ]

def create_race(policy=None, track="abyss", step_size=0.1):
    # Create the multi-agent environment for N karts.
    env = STKRaceMultiEnv(agents=agents_specs, track=track, render_mode="human", num_kart=MAX_TEAMS, laps=1, step_size=step_size)

    agents = [policy if policy is not None else Agent4(env, path_lookahead=3)]
    names = []

    for i in range(MAX_TEAMS):
        names.append(agents[i].name)
        agents_specs[i].name = agents[i].name
        agents_specs[i].kart = agents[i].name
    return env, agents, names


def single_race(env, agents, names, output, seed=None, max_steps=MAX_STEPS, actor=False, delay=0):
    obs, _ = env.reset(seed=seed)
    recorder = DatasetRecorder() if output is not None else None
    expert = Agent4(env, path_lookahead=3) if actor and recorder is not None else None
    if recorder is not None:
        recorder.start_episode(obs["0"], metadata={
            "track": env.current_track,
            "agent": names[0],
            "num_kart": MAX_TEAMS,
            "laps": 1,
            "max_steps": max_steps,
            "seed": seed,
        })
    for step in range(max_steps):
        # Expert errors must not silently become demonstration labels.
        label = expert.choose_action(deepcopy(obs["0"])) if expert is not None else None
        action = agents[0].choose_action(obs["0"])
        obs, _, terminated, truncated, info = env.step({"0": action})
        time.sleep(delay)  # slow down the display
        truncated = truncated or (step + 1 == max_steps and not terminated)
        if recorder is not None:
            recorder.record_step(
                action, info["reward"]["0"], obs["0"], terminated, truncated, expert_action=label
            )
        if terminated or truncated:
            break

    print(f"Finished={terminated}, truncated={truncated}, steps={step + 1}")
    if recorder is not None:
        recorder.save_episode(output)
        print(f"Saved {step + 1} steps to {output}")

def load_rlpd(checkpoint, config):
    # rebuild the RLPD agent with the run's own settings
    from agents.rlpd_agent import RLPDAgent
    from utils.observation_encoder import ObservationEncoder
    from utils.observation_normalizer import ObservationNormalizer
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    normalizer = ObservationNormalizer.load(checkpoint.parent / 'normalizer.npz', encoder)
    policy = RLPDAgent(config, encoder, normalizer, config['seed'])
    policy.load(checkpoint)
    return policy


def main_loop(output, checkpoint=None, seed=None, device='auto', track=None, delay=0):
    policy = None
    track_override = track
    track, step_size, max_steps = 'abyss', 0.1, MAX_STEPS
    if checkpoint is not None:
        config_path = checkpoint.parent / 'config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if 'rlpd' in config:  # RLPD run folder
            policy = load_rlpd(checkpoint, config)
        else:
            from agents.dagger_actor import ActorPolicy
            policy = ActorPolicy.load(checkpoint, device=device)
        if config:
            track = config['train_tracks'][0] if 'train_tracks' in config else config['track']
            step_size, max_steps = config['step_size'], config['max_steps']
    if track_override is not None:
        track = track_override
    print(f'Policy={checkpoint or "Team4"}; track={track}, max_steps={max_steps}')
    env, agents, names = create_race(policy, track, step_size)
    try:
        # print("Observation space:", env.observation_space)
        # print("Action space:", env.action_space)
        single_race(env, agents, names, output, seed, max_steps, actor=checkpoint is not None, delay=delay)
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional recording destination")
    parser.add_argument("--checkpoint", type=Path, help="Saved actor .pt; otherwise drive with Team4")
    parser.add_argument("--track", help="Override the experiment track, e.g. minigolf")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument("--delay", type=float, default=0, help="Seconds to pause per step; 0.1 is about real speed")
    args = parser.parse_args()
    main_loop(**vars(args))
