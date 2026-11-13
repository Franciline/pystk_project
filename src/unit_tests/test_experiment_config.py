"""Run: PYTHONPATH=src python src/unit_tests/test_experiment_config.py."""
from omegaconf import OmegaConf
from utils.experiment_config import CONFIG_PATH, load_config


if __name__ == '__main__':
    profiles = OmegaConf.load(CONFIG_PATH).profiles
    for name in profiles:
        c = OmegaConf.to_container(load_config(profile=name), resolve=True)
        assert (c['method'] == 'dagger') == ('actor' in c)
        assert (c['method'] == 'rlpd') == ('rlpd' in c)
        assert c['evaluation']['seeds'] == list(range(10000, 10005))
        if name.startswith('multimap'):
            assert len(c['train_tracks']) == 5 and len(c['unseen_tracks']) == 3
            assert c['normalize_observations']
        if 'initialization' in c and c['initialization']['mode'] == 'load_offline':
            assert c['initialization']['dataset'] == c['offline']['dataset']
    changed = load_config(profile='multimap_dagger', overrides=['seed=2000', 'offline.episodes_per_epsilon=4'])
    assert changed.seed == 2000 and changed.initialization.episodes_per_epsilon == 4
    assert load_config(profile='multimap_dagger').seed == 1000
    try:
        load_config(profile='multimap_rlpd', method='dagger')
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong-method profile was accepted')
    print('All profiles resolve, remain isolated, preserve overrides and reject the wrong method')
