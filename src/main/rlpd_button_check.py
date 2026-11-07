"""test a saved RLPD policy with some buttons forced off.
Run: .venv/bin/python src/main/rlpd_button_check.py datasets/rlpd_runs/t01_7act
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.rlpd_agent import BUTTONS, RLPDAgent
from main.DAgger import to_action
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.dagger_evaluation import evaluate_policy, append_metrics
from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer


class ButtonsOff:
    """same policy, but the listed buttons are never pressed"""

    def __init__(self, agent, off):
        self.agent, self.off = agent, off

    def choose_action(self, obs):
        action = self.agent.choose_action(obs)
        for key in self.off:
            action[key] = False
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--checkpoint', default='agent_last.pt')
    args = parser.parse_args()

    # rebuild the agent with the run's own settings
    config = json.loads((args.run_dir / 'config.json').read_text())
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    normalizer = ObservationNormalizer.load(args.run_dir / 'normalizer.npz', encoder)
    agent = RLPDAgent(config, encoder, normalizer, config['seed'])
    agent.load(args.run_dir / args.checkpoint)

    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=config['track'],
                     num_kart=1, laps=1, step_size=config['step_size'])
    try:
        for off in ([], ['rescue'], list(BUTTONS)):
            _, summary = evaluate_policy(env, ButtonsOff(agent, off), config['evaluation']['seeds'],
                                         config['max_steps'], to_action)
            name = '+'.join(off) or 'none'
            append_metrics(args.run_dir / 'button_check.csv',
                           [dict(checkpoint=args.checkpoint, off=name, **summary)])
            print(f"off={name}: completion={summary['completion_rate']:.0%}, "
                  f"finish steps={summary['finish_steps_mean']}", flush=True)
    finally:
        env.close()