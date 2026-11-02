"""collect an offline rlpd dataset: team agents with epsilon random actions, one lap each"""

import sys,os,random
from pathlib import Path
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))
# Team2 loads its config relative to the working directory, as if run from src/main.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from agents.team1.agent1 import Agent1
from agents.team2.agent2 import Agent2
from agents.team3.agent3 import Agent3
from agents.team4.agent4 import Agent4
from agents.team5.agent5 import Agent5

from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.dataset_recorder import DatasetRecorder

STEP_SIZE = 0.1  # seconds per env step (pystk2 default); lap_time = n_steps * STEP_SIZE

TRACK = "abyss"
AGENTS = {"team1": Agent1, "team2": Agent2, "team3": Agent3, "team4": Agent4, "team5": Agent5}
EPSILONS = [0.0, 0.1, 0.3]
SEEDS = range(10)
MAX_STEPS = 1000  # timeout: episode is truncated after this many steps
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "datasets" / "rlpd"

BUTTONS = ("brake", "drift", "fire", "nitro", "rescue")


def to_action(action):
    "flatten an action dict to python scalars clipped to the action space"
    first = lambda key: np.ravel(action[key])[0]
    return {
        "acceleration": float(np.clip(first("acceleration"), 0, 1)),
        "steer": float(np.clip(first("steer"),-1,1)),
        **{key: int(first(key) > 0) for key in BUTTONS}
    }

class EpsilonAgent():
    "execute a random action with proba epsilon, otherwise the agent's action"

    def __init__(self, agent, epsilon, action_space, rng):
        self.agent, self.epsilon = agent, epsilon
        self.action_space, self.rng = action_space, rng

    def choose_action(self, obs):
        "return (executed action, agent's action); store the executed one"
        expert = to_action(self.agent.choose_action(obs))
        # draw every step even at epsilon=0 so the rng stream depends only on the seed 

        if self.rng.random() >= self.epsilon:
            return expert, expert

        action = to_action(self.action_space.sample())
        action["rescue"] = 0
        return action, expert

def seed_everything(seed,env):
    """seed every rng and reset env so (agent, epsilon, seed) replays one exact episode.
    call before building the agent because the constructor might use global rng. 
    returns the first obs and the epsilon rng. """
    
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    return obs, np.random.default_rng(seed)

def run_episode(env, name, epsilon, seed, output):
    """run one lap of agent 'name' with epsilon mixing and save it to output"""
    obs, rng = seed_everything(seed,env)
    agent = EpsilonAgent(AGENTS[name](env, path_lookahead=3), epsilon, env.action_space, rng)
    recorder = DatasetRecorder()
    recorder.start_episode(obs, metadata={
            "agent": name, "epsilon": epsilon, "seed": seed, "track": env.current_track,
            "max_steps": MAX_STEPS, "step_size": STEP_SIZE,
        })    
    for step in range(MAX_STEPS):
        action, expert = agent.choose_action(obs)
        obs, reward, terminated, _, _ = env.step(action)
        truncated = step + 1 == MAX_STEPS and not terminated
        recorder.record_step(action, reward, obs, terminated, truncated, expert_action=expert)
        if terminated or truncated:
            break
    n_steps = step + 1
    recorder.episode["metadata"].update(
        finished=terminated, n_steps=n_steps, lap_time= round(n_steps * STEP_SIZE,2) if terminated else None
    )
    recorder.save_episode(output)
    return terminated, n_steps

def main():
    env = STKRaceEnv(agent=AgentSpec(rank_start=0), track=TRACK, num_kart=1, laps=1, step_size=STEP_SIZE)
    try:
        for name in AGENTS:
            for epsilon in EPSILONS:
                for seed in SEEDS:
                    output = OUTPUT_DIR / TRACK / f"{name}_eps{epsilon}_seed{seed}.pkl.gz"
                    if output.exists():
                        continue
                    try:
                        finished, n_steps = run_episode(env, name, epsilon, seed, output)
                    except Exception as error:
                        # One crashing agent must not kill a long run; nothing is saved, so a rerun retries it.
                        print(f"FAILED {output.name}: {error!r}", flush=True)
                        continue
                    status = "finished" if finished else "timeout"
                    print(f"{output.name}: {status} in {n_steps} steps", flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    main()
