"""RLPD training: SAC with 50/50 offline expert / online batches, evaluated like DAgger
Run: uv run src/main/RLPD_opti.py [--config path] [key=value overrides]
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.rlpd_agent_opti import RLPDAgent, vector_to_action, materialize_logs
from utils.rlpd_replay_opti import SymmetricSampler
from utils.rlpd_dataset_opti import prepare_offline
from main.DAgger import to_action
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
from utils.dagger_evaluation import is_improvement, append_metrics
from utils.dataset_recorder import DatasetRecorder
from utils.multitrack_rlpd_opti import (train_tracks,next_track,evaluate_tracks,track_step_limit,
    prepare_config,save_expert_reference,MapCollection,report_collection)

CONFIG_PATH = Path(__file__).with_name('full5maps50k_rlpd_opti.yaml')
FIELDS = ('obs', 'action', 'reward', 'next_obs', 'terminated')



# the rlpd loop: the agent drives (picks a slightly random action and the game moves on step)
# record the transition into the online buffer
# and learn with the configured UTD ratio

def run_rlpd(config):
    from utils.multitrack_rlpd_opti import start_workers, stop_workers
    torch.set_num_threads(int(config.get('optimization', {}).get('torch_threads', 2)))
    preparation_started = time.perf_counter()
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
    encoder, normalizer, offline, offline_summary = prepare_offline(config, output_dir)
    append_metrics(output_dir/'offline_summary.csv', offline_summary)
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

    sampler = SymmetricSampler(offline, online, cfg['batch_size'], cfg['utd_ratio'], rng, agent.device)

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
    preparation_seconds = time.perf_counter() - preparation_started
    append_metrics(output_dir/'preparation_timing.csv', [dict(seconds=preparation_seconds)])
    timing = dict(simulation_seconds=0., encoding_seconds=0., replay_seconds=0.,
                  update_seconds=0., saving_seconds=0.)
    interval_started = time.perf_counter()
    try:
        start_workers(config)
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
                stage_started = time.perf_counter()
                action = agent.act(obs)            # random sample: exploration
                executed = to_action(vector_to_action(action,agent.learned))
                raw_next, reward, terminated, truncated, _ = env.step(executed)
                truncated = bool(truncated or (step+1==max_steps and not terminated))
                recorder.record_step(executed,reward,raw_next,terminated,truncated)
                timing['simulation_seconds'] += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                next_obs = encode(raw_next)
                timing['encoding_seconds'] += time.perf_counter() - stage_started
                for key, value in zip(FIELDS, (obs, action, reward, next_obs, float(terminated))):
                    online[key][count] = value
                count, env_step = count + 1, env_step + 1
                episode_return += reward
                obs = next_obs
                if count >= half:                  # train once online data can fill half a batch
                    stage_started = time.perf_counter()
                    batches = sampler(count)
                    timing['replay_seconds'] += time.perf_counter() - stage_started
                    stage_started = time.perf_counter()
                    logs = agent.update(batches, defer_logs=True)
                    timing['update_seconds'] += time.perf_counter() - stage_started
                if terminated or truncated:
                    break

            collection.record(track,step+1)
            interval_steps[track]+=step+1
            interval_episodes[track]+=1
            diagnostic_summary = recorder.diagnostics.summary() if recorder.diagnostics else {}
            stage_started = time.perf_counter()
            logs = materialize_logs(logs)
            recorder.save_episode(output_dir/'online_trajectories'/f'episode_{episode+1}.pkl.gz')
            timing['saving_seconds'] += time.perf_counter() - stage_started
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
                stage_started = time.perf_counter()
                agent.evaluation_checkpoint = output_dir/'agent_last.pt'
                agent.save(agent.evaluation_checkpoint)
                timing['saving_seconds'] += time.perf_counter() - stage_started
                evaluation_started = time.perf_counter()
                rows,map_summaries,summary = evaluate_tracks(env,agent,config,
                    output_dir/'evaluation_episodes'/f'steps_{env_step}',to_action)
                evaluation_seconds = time.perf_counter() - evaluation_started
                append_metrics(output_dir/'timings.csv', [dict(env_steps=env_step,
                    interval_seconds=time.perf_counter()-interval_started,
                    evaluation_seconds=evaluation_seconds, **timing)])
                timing = {key:0. for key in timing}
                interval_started = time.perf_counter()
                append_metrics(output_dir/'evaluation_by_track.csv',
                    [dict(env_steps=env_step,**s) for s in map_summaries])
                improved = is_improvement(summary, best, ev['min_steps_improvement'])
                append_metrics(output_dir / 'evaluation_episodes.csv',
                               [dict(env_steps=env_step, **r) for r in rows])
                append_metrics(output_dir / 'evaluation.csv',
                               [dict(env_steps=env_step, train_episodes=episode, **summary, improved=improved)])
                print(f"Evaluation @ {env_step}: completion={summary['completion_rate']:.0%}, "
                      f"finish steps={summary['finish_steps_mean']}", flush=True)
                if improved:
                    best = summary
                    shutil.copy2(agent.evaluation_checkpoint, output_dir/'agent_best.pt')
                    with (output_dir / 'best_evaluation.json').open('w') as f:
                        json.dump(dict(env_steps=env_step,selection_tracks=tracks,
                            selection_rule='Train-map completion, then macro finishing steps; unseen excluded',
                            **summary),f,indent=2)
    finally:
        stop_workers()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--experiment',type=Path,help='Shared multi-map overrides')
    args, overrides = parser.parse_known_args()
    # key=value overrides on the command line, e.g. total_env_steps=2000
    config = OmegaConf.merge(OmegaConf.load(args.config),
        OmegaConf.load(args.experiment) if args.experiment else {},OmegaConf.from_dotlist(overrides))
    run_rlpd(OmegaConf.to_container(config, resolve=True))
