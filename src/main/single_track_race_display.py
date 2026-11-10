"""Display a saved actor or Team4 for one lap, optionally recording."""

import argparse
import json
import random
import numpy as np
from copy import deepcopy
from importlib import import_module
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

def build_agent(team, env):
    # import only the chosen team: team2 needs src/main as working folder
    module = import_module(f"agents.team{team}.agent{team}")
    return getattr(module, f"Agent{team}")(env, path_lookahead=3)


def create_race(policy=None, track="abyss", step_size=0.1, team=4):
    # Create the multi-agent environment for N karts.
    env = STKRaceMultiEnv(agents=agents_specs, track=track, render_mode="human", num_kart=MAX_TEAMS, laps=1, step_size=step_size)

    agents = [policy if policy is not None else build_agent(team, env)]
    names = []

    for i in range(MAX_TEAMS):
        names.append(agents[i].name)
        agents_specs[i].name = agents[i].name
        agents_specs[i].kart = agents[i].name
    return env, agents, names


def single_race(env, agents, names, output, seed=None, max_steps=MAX_STEPS, actor=False,
                epsilon=0.0, hold_steps=1, delay=0):
    if not 0 <= epsilon <= 1 or hold_steps < 1:
        raise ValueError("Require epsilon in [0, 1] and hold_steps >= 1")
    if actor and epsilon > 0:
        raise ValueError("Random mixing is unavailable with checkpoint")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)
    use_random = False
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
            "epsilon": epsilon, "hold_steps": hold_steps,
        })
    for step in range(max_steps):
        # Expert errors must not silently become demonstration labels.
        label = expert.choose_action(deepcopy(obs["0"])) if expert is not None else None
        action = agents[0].choose_action(obs["0"])
        # Always query the expert above, including during random-policy blocks.
        if not actor:
            label = deepcopy(action)
        if step % hold_steps == 0:
            use_random = rng.random() < epsilon
        if use_random:
            sampled = env.action_space["0"].sample()
            action = {key: np.asarray(value).item() for key, value in sampled.items()}
            action["rescue"] = False
        # Uncomment to inspect executed action values.
        # print(f"Step {step}: {action}", flush=True)
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

def load_rlpd(checkpoint, config, no_normalization=False):
    if not no_normalization and not (checkpoint.parent / 'normalizer.npz').exists():
        from main.compare_generalization import load_rlpd as load_checkpoint
        return load_checkpoint(checkpoint, config['rlpd'].get('device', 'auto'))
    # rebuild the RLPD agent with the run's own settings
    from agents.rlpd_agent import RLPDAgent
    from utils.observation_encoder import ObservationEncoder
    from utils.observation_normalizer import ObservationNormalizer
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    if no_normalization:
        normalizer = ObservationNormalizer(encoder)
        normalizer.mean = np.zeros(encoder.output_size)
        normalizer.scale = np.ones(encoder.output_size)
    else:
        normalizer = ObservationNormalizer.load(checkpoint.parent / 'normalizer.npz', encoder)
    policy = RLPDAgent(config, encoder, normalizer, config['seed'])
    policy.load(checkpoint)
    return policy


def main_loop(output, checkpoint=None, seed=None, device='auto', track=None, delay=0, team=4, epsilon=0.0, hold_steps=1, no_normalization=False):
    if not 0 <= epsilon <= 1 or hold_steps < 1:
        raise ValueError('Require epsilon in [0, 1] and hold_steps >= 1')
    if checkpoint is not None and epsilon > 0:
        raise ValueError('Random mixing is only supported with the Team4 expert')
    policy = None
    track_override = track
    track, step_size, max_steps = 'abyss', 0.1, MAX_STEPS
    if checkpoint is not None:
        config_path = checkpoint.parent / 'config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if 'rlpd' in config:  # RLPD run folder
            policy = load_rlpd(checkpoint, config, no_normalization=no_normalization)
        else:
            from agents.dagger_actor import ActorPolicy
            policy = ActorPolicy.load(checkpoint, device=device)
        if no_normalization:
            # Identity statistics bypass scaling while retaining observation encoding.
            policy.normalizer.mean = np.zeros(policy.encoder.output_size)
            policy.normalizer.scale = np.ones(policy.encoder.output_size)
            print('Observation normalization disabled for this run', flush=True)
        if config:
            track = config['train_tracks'][0] if 'train_tracks' in config else config['track']
            step_size, max_steps = config['step_size'], config['max_steps']
    if track_override is not None:
        track = track_override
    print(f'Policy={checkpoint or f"Team{team}"}; track={track}, max_steps={max_steps}')
    env, agents, names = create_race(policy, track, step_size, team)
    try:
        # print("Observation space:", env.observation_space)
        # print("Action space:", env.action_space)
        single_race(env, agents, names, output, seed, max_steps, actor=checkpoint is not None,
                    epsilon=epsilon, hold_steps=hold_steps, delay=delay)
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional recording destination")
    parser.add_argument("--checkpoint", type=Path, help="Saved actor .pt; otherwise drive with Team4")
    parser.add_argument("--track", help="Override the experiment track, e.g. minigolf")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument("--no-normalization", action='store_true',
                        help="Feed encoded observations without scaling (checkpoint test only)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Random-policy probability per block")
    parser.add_argument("--hold-steps", type=int, default=1, help="Steps per policy-selection block")
    parser.add_argument("--delay", type=float, default=0, help="Seconds to pause per step; 0.1 is about real speed")
    parser.add_argument("--team", type=int, choices=range(1, 6), default=4, help="Team to drive when no checkpoint is given")
    args = parser.parse_args()
    main_loop(**vars(args))
