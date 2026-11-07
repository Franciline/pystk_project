"""RLPD training: SAC with 50/50 offline expert / online batches, evaluated like DAgger
Run: .venv/bin/python src/main/RLPD.py [--config path] [key=value overrides]
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.rlpd_agent import RLPDAgent, vector_to_action
from main.DAgger import to_action
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.dagger_evaluation import evaluate_policy, is_improvement, append_metrics
from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer
from utils.rlpd_dataset import load_episodes, drop_duplicates, build_offline_buffer

CONFIG_PATH = Path(__file__).with_name('rlpd_config.yaml')
FIELDS = ('obs', 'action', 'reward', 'next_obs', 'terminated')


def sample(buffer, count, size, rng):
    """size random transitions from the first `count` rows of a buffer."""
    index = rng.integers(0, count, size)
    return [buffer[key][index] for key in FIELDS]


# the rlpd loop: the agent drives (picks a slightly random action and the game moves on step)
# record the transition into the online buffer
# and learn with 3 training updates 

def run_rlpd(config):
    seed, max_steps = config['seed'], config['max_steps']
    cfg = config['rlpd']
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)              # batch sampling

    output_dir = Path(config['output_dir'])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'config.json').open('x') as f:  # refuse to mix two runs in one folder
        json.dump(config, f, indent=2)

    # --- offline expert buffer; normalizer fitted once on it, then frozen (like DAgger) ---
    print('Building offline buffer...', flush=True)
    episodes = drop_duplicates(load_episodes(agents=config['offline']['agents'],
                                             epsilons=config['offline']['epsilons'],
                                             tracks=[config['track']]))
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    normalizer = ObservationNormalizer(encoder).fit(o for e in episodes for o in e['observations'][:-1])
    normalizer.save(output_dir / 'normalizer.npz')
    offline = build_offline_buffer(episodes, encoder, normalizer, config['learned_actions'])
    del episodes                                   # free the raw dicts
    print(f"Offline buffer: {len(offline['reward'])} transitions", flush=True)

    agent = RLPDAgent(config, encoder, normalizer, seed)

    # --- online buffer, preallocated (one extra episode of room for the last overshoot) ---
    # ponytail: stores obs and next_obs separately (~0.8 GB at 50k steps); fine on one Mac.
    capacity = config['total_env_steps'] + max_steps
    action_size = len(agent.learned)
    online = {'obs': np.zeros((capacity, encoder.output_size), np.float32),
              'action': np.zeros((capacity, action_size), np.float32),
              'reward': np.zeros(capacity, np.float32),
              'next_obs': np.zeros((capacity, encoder.output_size), np.float32),
              'terminated': np.zeros(capacity, np.float32)}
    count = 0
    half = cfg['batch_size'] // 2

    def minibatch():
        """half expert, half own experience (RLPD symmetric sampling)."""
        off = sample(offline, len(offline['reward']), half, rng)
        on = sample(online, count, half, rng)
        return tuple(torch.as_tensor(np.concatenate([a, b])).to(agent.device) for a, b in zip(off, on))

    def encode(raw_obs):
        return normalizer.transform(encoder.encode(raw_obs))

    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=config['track'],
                     num_kart=1, laps=1, step_size=config['step_size'])
    env_step, episode, next_eval, best = 0, 0, config['eval_every'], None
    start = time.time()
    try:
        while env_step < config['total_env_steps']:
            episode_seed = config['online_seed_start'] + episode
            obs = encode(env.reset(seed=episode_seed)[0])
            episode_return, logs = 0.0, {}
            for step in range(max_steps):
                action = agent.act(obs)            # random sample: exploration
                raw_next, reward, terminated, _, _ = env.step(vector_to_action(action, agent.learned))
                next_obs = encode(raw_next)
                for key, value in zip(FIELDS, (obs, action, reward, next_obs, float(terminated))):
                    online[key][count] = value
                count, env_step = count + 1, env_step + 1
                episode_return += reward
                obs = next_obs
                if count >= half:                  # train once online data can fill half a batch
                    logs = agent.update([minibatch() for _ in range(cfg['utd_ratio'])])
                if terminated:
                    break

            row = {'episode': episode, 'seed': episode_seed, 'env_steps': env_step,
                   'n_steps': step + 1, 'finished': bool(terminated),
                   'return': round(episode_return, 2), **logs}
            append_metrics(output_dir / 'online_episodes.csv', [row])
            print(f"Episode {episode}: {'finished' if terminated else 'timeout'} in {step + 1} steps, "
                  f"return={episode_return:.1f}, env_steps={env_step}, {time.time() - start:.0f}s", flush=True)
            episode += 1

            # evaluate between episodes (evaluation resets the env)
            if env_step >= next_eval or env_step >= config['total_env_steps']:
                next_eval += config['eval_every']
                ev = config['evaluation']
                rows, summary = evaluate_policy(env, agent, ev['seeds'], max_steps, to_action)
                improved = is_improvement(summary, best, ev['min_steps_improvement'])
                append_metrics(output_dir / 'evaluation_episodes.csv',
                               [dict(env_steps=env_step, **r) for r in rows])
                append_metrics(output_dir / 'evaluation.csv',
                               [dict(env_steps=env_step, train_episodes=episode, **summary, improved=improved)])
                print(f"Evaluation @ {env_step}: completion={summary['completion_rate']:.0%}, "
                      f"finish steps={summary['finish_steps_mean']}", flush=True)
                agent.save(output_dir / 'agent_last.pt')
                if improved:
                    best = summary
                    agent.save(output_dir / 'agent_best.pt')
                    with (output_dir / 'best_evaluation.json').open('w') as f:
                        json.dump(dict(env_steps=env_step, **summary), f, indent=2)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    args, overrides = parser.parse_known_args()
    # key=value overrides on the command line, e.g. total_env_steps=2000
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    run_rlpd(OmegaConf.to_container(config, resolve=True))