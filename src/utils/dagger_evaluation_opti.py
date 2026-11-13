"""CPU-only evaluation workers, each with an isolated native simulator."""
from pathlib import Path

_env = None


def initialize_worker():
    import torch
    torch.set_num_threads(1)


def evaluate_map(job):
    global _env
    from multiprocessing.util import Finalize
    from agents.dagger_actor import ActorPolicy
    from main.DAgger import to_action
    from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
    from utils.dagger_evaluation import evaluate_policy
    from utils.multitrack import track_step_limit
    checkpoint, config, track, record_dir = job
    # Preserve the original serialized STK interface; direct native observations differ.
    if _env is None:
        _env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=track,
                          num_kart=1, laps=1, step_size=config['step_size'], use_subprocess=True)
        Finalize(None, _env.close, exitpriority=10)
    _env.default_track = track
    policy = ActorPolicy.load(checkpoint, device='cpu')
    detail, summary = evaluate_policy(_env, policy, config['evaluation']['seeds'],
        track_step_limit(config, track), to_action, diagnostics=config.get('diagnostics'),
        record_dir=Path(record_dir) / track
        if config.get('diagnostics', {}).get('save_evaluation_episodes', True) else None)
    return track, detail, summary
