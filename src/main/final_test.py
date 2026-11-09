""" the final tests for RLPD and DAgger: team4 and the best/last checkpoints of each run on 20 new race seeds.
for rlpd run: .venv/bin/python src/main/final_test.py --expert datasets/protocol/rlpd_seed1000 datasets/protocol/rlpd_seed2000 datasets/protocol/rlpd_seed3000
for dagger run: .venv/bin/python src/main/final_test.py datasets/protocol/dagger_offline_seed1000 datasets/protocol/dagger_offline_seed2000 datasets/protocol/dagger_offline_seed3000 
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.dagger_actor import ActorPolicy
from agents.rlpd_agent import RLPDAgent
from agents.team4.agent4 import Agent4
from main.DAgger import to_action
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.dagger_evaluation import evaluate_policy, append_metrics
from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer

PROTOCOL = {'track': 'abyss', 'step_size': 0.1, 'max_steps': 1000}  # same race for every policy
EXPERT_DIR = Path(__file__).resolve().parents[2] / 'datasets' / 'protocol' / 'expert_test'


def load_policy(run_dir, name):
    """rebuild a saved RLPD or DAgger policy from its run folder"""
    config = json.loads((run_dir / 'config.json').read_text())
    if any(config[key] != value for key, value in PROTOCOL.items()):
        raise ValueError(f'{run_dir} was not trained with the protocol settings')
    if 'rlpd' not in config:  # DAgger run
        return ActorPolicy.load(run_dir / f'actor_{name}.pt')
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    normalizer = ObservationNormalizer.load(run_dir / 'normalizer.npz', encoder)
    policy = RLPDAgent(config, encoder, normalizer, config['seed'])
    policy.load(run_dir / f'agent_{name}.pt')
    return policy


def test(env, out_dir, checkpoint, policy, seeds):
    """race one policy on every seed, save each race and the summary"""
    expert = (lambda env: Agent4(env, path_lookahead=3)) if policy is None else None  # fresh team4 per race
    rows, summary = evaluate_policy(env, policy, seeds, PROTOCOL['max_steps'], to_action,
                                    policy_factory=expert)
    append_metrics(out_dir / 'final_test_episodes.csv', [dict(checkpoint=checkpoint, **r) for r in rows])
    append_metrics(out_dir / 'final_test.csv', [dict(checkpoint=checkpoint, **summary)])
    print(f"{out_dir.name} {checkpoint}: completion={summary['completion_rate']:.0%}, "
          f"finish steps={summary['finish_steps_mean']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', type=Path, nargs='*')
    parser.add_argument('--expert', action='store_true', help='also test team4 (only one of us runs it)')
    parser.add_argument('--seed-start', type=int, default=20000)
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.runs))  # never used in training or validation

    out_dirs = args.run_dirs + ([EXPERT_DIR] if args.expert else [])
    for out_dir in out_dirs:  # stop before racing: a rerun would mix two tests in one file
        if (out_dir / 'final_test.csv').exists():
            raise FileExistsError(f'{out_dir} already tested')
    if args.expert:
        EXPERT_DIR.mkdir(parents=True, exist_ok=True)

    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=PROTOCOL['track'],
                     num_kart=1, laps=1, step_size=PROTOCOL['step_size'])
    try:
        if args.expert:
            test(env, EXPERT_DIR, 'team4', None, seeds)
        for run_dir in args.run_dirs:
            for name in ('best', 'last'):
                test(env, run_dir, name, load_policy(run_dir, name), seeds)
    finally:
        env.close()