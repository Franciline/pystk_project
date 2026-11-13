"""Resolve one explicit method profile, or load an external standalone YAML."""
from pathlib import Path
from omegaconf import OmegaConf

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'main/experiments.yaml'


def load_config(path=CONFIG_PATH, profile=None, *, overrides=(), experiment=None, method=None):
    catalog = OmegaConf.load(path)
    if 'profiles' in catalog:
        if profile not in catalog.profiles:
            raise ValueError(f'Choose a config profile from {list(catalog.profiles)}')
        # Detach nodes so ${offline.*} resolves against the selected profile root.
        config = OmegaConf.merge(OmegaConf.to_container(catalog.common, resolve=False),
                                 OmegaConf.to_container(catalog.profiles[profile], resolve=False))
    else:
        config = catalog
    config = OmegaConf.merge(config, OmegaConf.load(experiment) if experiment else {},
                             OmegaConf.from_dotlist(overrides))
    if method is not None and config.get('method', method) != method:
        raise ValueError(f'This entry point requires a {method} profile')
    selected_method = config.get('method', method)
    if selected_method == 'dagger' and 'rlpd' in config:
        raise ValueError('A DAgger profile must not contain an RLPD section')
    if selected_method == 'rlpd' and 'actor' in config:
        raise ValueError('An RLPD profile must not contain a DAgger actor section')
    return config
