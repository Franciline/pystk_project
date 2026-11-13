"""Independent CPU RLPD evaluation workers using the original serialized simulator interface."""
from copy import deepcopy
from pathlib import Path

_env = None


def initialize_worker():
    import torch
    torch.set_num_threads(1)


def load_policy(checkpoint, config):
    import torch
    from agents.rlpd_agent import RLPDAgent
    from utils.observation_encoder import ObservationEncoder
    from utils.observation_normalizer import ObservationNormalizer
    config = deepcopy(config)
    config['rlpd']['device'] = 'cpu'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    if state['learned'] != config['learned_actions']:
        raise ValueError('Checkpoint action layout differs from evaluation config')
    normalizer = ObservationNormalizer(encoder)
    normalizer.mean = state['normalizer_mean'].numpy().copy()
    normalizer.scale = state['normalizer_scale'].numpy().copy()
    policy = RLPDAgent(config, encoder, normalizer, config['seed'])
    policy.actor.load_state_dict(state['actor'])
    policy.actor.eval()
    return policy


def evaluate_map(job):
    global _env
    from multiprocessing.util import Finalize
    from main.DAgger import to_action
    from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
    from utils.dagger_evaluation import evaluate_policy
    from utils.multitrack import track_step_limit
    checkpoint, config, track, record_dir = job
    if _env is None:
        _env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=track,
            num_kart=1, laps=1, step_size=config['step_size'], use_subprocess=True)
        Finalize(None, _env.close, exitpriority=10)
    _env.default_track = track
    policy = load_policy(checkpoint, config)
    rows, summary = evaluate_policy(_env, policy, config['evaluation']['seeds'],
        track_step_limit(config, track), to_action, diagnostics=config.get('diagnostics'),
        record_dir=Path(record_dir)/track
        if config.get('diagnostics', {}).get('save_evaluation_episodes', True) else None)
    return track, rows, summary
