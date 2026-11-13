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
from utils.dataset_recorder import DatasetRecorder
from utils.multitrack import (train_tracks,next_track,evaluate_tracks,track_step_limit,
    prepare_config,save_expert_reference,MapCollection,report_collection)

from utils.experiment_config import CONFIG_PATH, load_config
FIELDS = ('obs', 'action', 'reward', 'next_obs', 'terminated')


def sample(buffer, count, size, rng):
    """size random transitions from the first `count` rows of a buffer."""
    index = rng.integers(0, count, size)
    return [buffer[key][index] for key in FIELDS]


# the rlpd loop: the agent drives (picks a slightly random action and the game moves on step)
# record the transition into the online buffer
# and learn with 3 training updates 

def run_rlpd(config):
    config=prepare_config(config)
    seed=config['seed']
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

    tracks = train_tracks(config)
    # Raw encoded inputs by default; optional fitted normalization is explicit.
    print('Building offline buffer...', flush=True)
    dataset = Path(config['offline'].get('dataset','datasets/rlpd'))
    if not dataset.is_absolute():
        dataset = Path(__file__).resolve().parents[2]/dataset
    episodes = drop_duplicates(load_episodes(root=dataset,agents=config['offline']['agents'],
                                             epsilons=config['offline']['epsilons'],
                                             tracks=tracks))
    missing = set(tracks)-{e['metadata']['track'] for e in episodes}
    if missing:
        raise ValueError(f'Missing offline data for tracks: {sorted(missing)}; collect it first')
    append_metrics(output_dir/'offline_summary.csv',[dict(track=track,
        episodes=sum(e['metadata']['track']==track for e in episodes),
        transitions=sum(len(e['actions']) for e in episodes if e['metadata']['track']==track)) for track in tracks])
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    normalizer = (ObservationNormalizer(encoder).fit(o for e in episodes for o in e['observations'][:-1])
                  if config.get('normalize_observations',False) else ObservationNormalizer.identity(encoder))
    normalizer.save(output_dir / 'normalizer.npz')
    offline = build_offline_buffer(episodes, encoder, normalizer, config['learned_actions'])
    del episodes                                   # free the raw dicts
    print(f"Offline buffer: {len(offline['reward'])} transitions", flush=True)

    agent = RLPDAgent(config, encoder, normalizer, seed)

    # --- online buffer, preallocated (one extra episode of room for the last overshoot) ---
    # ponytail: stores obs and next_obs separately (~0.8 GB at 50k steps); fine on one Mac.
    capacity = config['total_env_steps'] + max(track_step_limit(config,t) for t in tracks)
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

    save_expert_reference(config,output_dir)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=tracks[0],
                     num_kart=1, laps=1, step_size=config['step_size'])
    env_step, episode, next_eval, best = 0, 0, config['eval_every'], None
    collection=MapCollection(tracks,config["online_seed_start"])
    interval_steps={track:0 for track in tracks}
    interval_episodes={track:0 for track in tracks}
    start = time.time()
    try:
        while env_step < config['total_env_steps']:
            track = next_track(tracks,collection.steps)
            episode_seed=collection.next_seed(track)
            max_steps=track_step_limit(config,track)
            env.default_track = track
            raw_obs = env.reset(seed=episode_seed)[0]
            recorder = DatasetRecorder(env,config.get('diagnostics'),expert_labels=False)
            recorder.start_episode(raw_obs,metadata=dict(track=track,seed=episode_seed,
                agent='rlpd',step_size=config['step_size'],max_steps=max_steps,laps=1,num_kart=1,role='online_training'))
            obs = encode(raw_obs)
            episode_return, logs = 0.0, dict(critic_loss=None,actor_loss=None,
                temperature=float(agent.log_alpha.exp().item()))
            for step in range(max_steps):
                action = agent.act(obs)            # random sample: exploration
                executed = to_action(vector_to_action(action,agent.learned))
                raw_next, reward, terminated, truncated, _ = env.step(executed)
                truncated = bool(truncated or (step+1==max_steps and not terminated))
                recorder.record_step(executed,reward,raw_next,terminated,truncated)
                next_obs = encode(raw_next)
                for key, value in zip(FIELDS, (obs, action, reward, next_obs, float(terminated))):
                    online[key][count] = value
                count, env_step = count + 1, env_step + 1
                episode_return += reward
                obs = next_obs
                if count >= half:                  # train once online data can fill half a batch
                    logs = agent.update([minibatch() for _ in range(cfg['utd_ratio'])])
                if terminated or truncated:
                    break

            collection.record(track,step+1)
            interval_steps[track]+=step+1
            interval_episodes[track]+=1
            diagnostic_summary = recorder.diagnostics.summary() if recorder.diagnostics else {}
            recorder.save_episode(output_dir/'online_trajectories'/f'episode_{episode+1}.pkl.gz')
            row = {'episode': episode, 'seed': episode_seed, 'track': track,'env_steps': env_step,
                   'n_steps': step + 1, 'finished': bool(terminated),
                   'return': round(episode_return, 2), 'truncated':truncated,**diagnostic_summary,**logs}
            append_metrics(output_dir / 'online_episodes.csv', [row])
            print(f"Episode {episode} ({track}): {'finished' if terminated else 'timeout'} in {step + 1} steps, "
                  f"return={episode_return:.1f}, env_steps={env_step}, {time.time() - start:.0f}s", flush=True)
            episode += 1

            # evaluate between episodes (evaluation resets the env)
            if env_step >= next_eval or env_step >= config['total_env_steps']:
                next_eval=(env_step//config['eval_every']+1)*config['eval_every']
                report_collection(output_dir/'collection_by_track.csv',
                    collection.summary(interval_episodes,interval_steps,env_steps=env_step))
                interval_steps={track:0 for track in tracks}
                interval_episodes={track:0 for track in tracks}
                ev = config['evaluation']
                rows,map_summaries,summary = evaluate_tracks(env,agent,config,
                    output_dir/'evaluation_episodes'/f'steps_{env_step}',to_action)
                append_metrics(output_dir/'evaluation_by_track.csv',
                    [dict(env_steps=env_step,**s) for s in map_summaries])
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
                        json.dump(dict(env_steps=env_step,selection_tracks=tracks,
                            selection_rule='Train-map completion, then macro finishing steps; unseen excluded',
                            **summary),f,indent=2)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--profile', default='singlemap_rlpd')
    parser.add_argument('--experiment',type=Path,help='Shared multi-map overrides')
    args, overrides = parser.parse_known_args()
    # key=value overrides on the command line, e.g. total_env_steps=2000
    config = load_config(args.config, args.profile, experiment=args.experiment, overrides=overrides, method='rlpd')
    run_rlpd(OmegaConf.to_container(config, resolve=True))