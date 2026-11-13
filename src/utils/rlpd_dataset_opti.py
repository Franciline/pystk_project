"""Content-addressed, read-only offline replay cache shared across training seeds."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer
from utils.rlpd_dataset import load_episodes, drop_duplicates, build_offline_buffer
from utils.multitrack import train_tracks

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('obs', 'action', 'reward', 'next_obs', 'terminated')


def prepare_offline(config, output_dir):
    dataset = ROOT / config['offline']['dataset']
    sources = sorted(dataset.rglob('*.pkl.gz'))
    if not sources:
        raise ValueError(f'No offline episodes in {dataset}')
    encoder = ObservationEncoder(config['encoder']['num_paths'], config['encoder']['num_items'])
    signature = dict(layout=1, encoder=ObservationNormalizer(encoder)._signature(),
        code={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
              for name in ('observation_encoder.py', 'observation_normalizer.py', 'rlpd_dataset.py')},
        tracks=train_tracks(config), agents=config['offline']['agents'], epsilons=config['offline']['epsilons'],
        learned=config['learned_actions'], normalize=config.get('normalize_observations', False),
        sources=[(str(path.relative_to(dataset)), hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources])
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_root = ROOT / config['optimization']['offline_cache']
    cache = cache_root / digest
    if not cache.exists():
        print('Building encoded offline replay cache...', flush=True)
        episodes = drop_duplicates(load_episodes(root=dataset, agents=config['offline']['agents'],
            epsilons=config['offline']['epsilons'], tracks=train_tracks(config)))
        missing = set(train_tracks(config)) - {e['metadata']['track'] for e in episodes}
        if missing:
            raise ValueError(f'Missing offline data for tracks: {sorted(missing)}')
        normalizer = (ObservationNormalizer(encoder).fit(o for e in episodes for o in e['observations'][:-1])
            if config.get('normalize_observations', False) else ObservationNormalizer.identity(encoder))
        buffer = build_offline_buffer(episodes, encoder, normalizer, config['learned_actions'])
        summary = [dict(track=t, episodes=sum(e['metadata']['track']==t for e in episodes),
            transitions=sum(len(e['actions']) for e in episodes if e['metadata']['track']==t))
            for t in train_tracks(config)]
        cache_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='building-', dir=cache_root) as temporary:
            staging = Path(temporary) / 'cache'
            staging.mkdir()
            for key in FIELDS:
                np.save(staging / f'{key}.npy', buffer[key], allow_pickle=False)
            normalizer.save(staging / 'normalizer.npz')
            (staging / 'manifest.json').write_text(json.dumps(dict(signature=signature, summary=summary), indent=2))
            # Never expose a partial cache to another run. Identical completed caches are reusable.
            if not cache.exists():
                try:
                    staging.rename(cache)
                except OSError:
                    if not cache.exists():
                        raise
    print('Offline replay cache:', cache, flush=True)
    manifest = json.loads((cache / 'manifest.json').read_text())
    # JSON turns source tuples into lists; compare canonical JSON forms.
    if json.dumps(manifest['signature'], sort_keys=True) != json.dumps(signature, sort_keys=True):
        raise ValueError('Offline replay cache signature mismatch')
    normalizer = ObservationNormalizer.load(cache / 'normalizer.npz', encoder)
    normalizer.save(Path(output_dir) / 'normalizer.npz')
    buffer = {key: np.load(cache / f'{key}.npy', mmap_mode='r', allow_pickle=False) for key in FIELDS}
    size = len(buffer['reward'])
    expected = {'obs': (size, encoder.output_size), 'next_obs': (size, encoder.output_size),
                'action': (size, len(config['learned_actions'])), 'reward': (size,), 'terminated': (size,)}
    if size < 1 or any(buffer[k].shape != expected[k] or buffer[k].dtype != np.float32 for k in FIELDS):
        raise ValueError('Invalid offline replay cache arrays')
    return encoder, normalizer, buffer, manifest['summary']
