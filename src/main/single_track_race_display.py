"""Display and record one Team4 kart completing one lap."""

import argparse
import sys, os
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

def create_race():
    # Create the multi-agent environment for N karts.
    env = STKRaceMultiEnv(agents=agents_specs, track="abyss", render_mode="human", num_kart=MAX_TEAMS, laps=1)

    agents = [Agent4(env, path_lookahead=3)]
    names = []

    for i in range(MAX_TEAMS):
        names.append(agents[i].name)
        agents_specs[i].name = agents[i].name
        agents_specs[i].kart = agents[i].name
    return env, agents, names


def single_race(env, agents, names, output):
    obs, _ = env.reset()
    recorder = DatasetRecorder()
    recorder.start_episode(obs["0"], metadata={
        "track": env.current_track,
        "agent": names[0],
        "num_kart": MAX_TEAMS,
        "laps": 1,
        "max_steps": MAX_STEPS,
    })
    for step in range(MAX_STEPS):
        # Expert errors must not silently become demonstration labels.
        action = agents[0].choose_action(obs["0"])
        obs, _, terminated, truncated, info = env.step({"0": action})
        truncated = truncated or (step + 1 == MAX_STEPS and not terminated)
        recorder.record_step(
            action, info["reward"]["0"], obs["0"], terminated, truncated
        )
        if terminated or truncated:
            break

    recorder.save_episode(output)
    print(f"Saved {step + 1} steps to {output}")

def main_loop(output):
    env, agents, names = create_race()
    try:
        # print("Observation space:", env.observation_space)
        # print("Action space:", env.action_space)
        single_race(env, agents, names, output)
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main_loop(parser.parse_args().output)
