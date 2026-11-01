import gymnasium as gym
from pystk2_gymnasium.envs import AgentSpec
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))
from utils.track_path_wrapper import plot_agent_path_with_track


def main():
    track_name = "minigolf"
    agent = AgentSpec(name="Player", use_ai=True)
    env = gym.make("supertuxkart/full-v0", render_mode="human", agent=agent, track=track_name, num_kart=1, laps=1)

    try:
        plot_agent_path_with_track(agent, env, track_name)
    finally:
        env.close()


if __name__ == "__main__":
    main()
