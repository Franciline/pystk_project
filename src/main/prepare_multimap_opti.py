"""Extend the existing six-map expert reference and three-map offline data without changing either."""
import argparse
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main.benchmark_expert import run_benchmark, prepare_protocol
from main.collect_offline import collect_dataset
from utils.experiment_config import CONFIG_PATH, load_config
from utils.episode_deduplication import drop_duplicates, deduplication_report
from utils.multitrack import evaluation_tracks

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open() as file:
        return list(csv.DictReader(file))


def prepare_expert(config):
    target = ROOT / config['track_timeouts']['directory']
    if (target / 'track_timeouts.json').exists():
        return prepare_protocol(config)
    source = ROOT / config['optimization']['reuse_expert']
    old = json.loads((source / 'track_timeouts.json').read_text())
    seeds = list(range(config['expert_benchmark']['seed_start'],
                       config['expert_benchmark']['seed_start'] + config['expert_benchmark']['runs']))
    for key, value in dict(step_size=config['step_size'], seeds=seeds,
            margin_steps=config['track_timeouts']['margin_steps'],
            benchmark_max_steps=config['track_timeouts']['benchmark_max_steps']).items():
        if old['protocol'][key] != value:
            raise ValueError(f'Reused expert calibration differs: {key}')
    tracks = evaluation_tracks(config)
    missing = [t for t in tracks if t not in old['protocol']['tracks']]
    extra = target.with_name(target.name + '_additions')
    if missing:
        if not (extra / 'summary.csv').exists():
            c = deepcopy(config)
            c.update(train_tracks=missing, unseen_tracks=[],
                     max_steps=config['track_timeouts']['benchmark_max_steps'], max_steps_by_track={})
            run_benchmark(c, extra, seeds)
        recorded = json.loads((extra / 'config.json').read_text())
        if recorded['seeds'] != seeds or evaluation_tracks(recorded['task_config']) != missing:
            raise ValueError('New expert reference does not match requested maps/seeds')
        summaries = read_csv(extra / 'summary.csv')
        if {r['track'] for r in summaries} != set(missing):
            raise ValueError('Incomplete new expert calibration; finish it in a fresh directory')
    stage = target.with_name(target.name + '_building')
    if stage.exists() or target.exists():
        raise ValueError(f'Incomplete preparation exists: {stage} or {target}; inspect before retrying')
    shutil.copytree(source, stage)
    summary = [r for r in read_csv(source / 'summary.csv') if r['track'] in tracks]
    episodes = [r for r in read_csv(source / 'episodes.csv') if r['track'] in tracks]
    if missing:
        summary += read_csv(extra / 'summary.csv')
        episodes += read_csv(extra / 'episodes.csv')
        for directory in ('track_geometry', 'episodes'):
            if (extra / directory).exists():
                shutil.copytree(extra / directory, stage / directory, dirs_exist_ok=True)
    limits = {}; details = {}
    for track in tracks:
        rows = [r for r in episodes if r['track'] == track]
        finished = [int(r['n_steps']) for r in rows if r['terminated'].lower() == 'true']
        if len(rows) != len(seeds) or not finished:
            raise ValueError(f'Incomplete expert reference for {track}')
        limits[track] = max(finished) + config['track_timeouts']['margin_steps']
        details[track] = dict(max_expert_finishing_steps=max(finished), completed=len(finished), episodes=len(rows))
    write_csv(stage / 'summary.csv', summary)
    write_csv(stage / 'episodes.csv', episodes)
    signature = dict(tracks=tracks, step_size=config['step_size'], seeds=seeds,
        margin_steps=config['track_timeouts']['margin_steps'],
        benchmark_max_steps=config['track_timeouts']['benchmark_max_steps'])
    (stage / 'config.json').write_text(json.dumps(dict(task_config=config, seeds=seeds, expert='team4'), indent=2))
    (stage / 'track_timeouts.json').write_text(json.dumps(dict(protocol=signature,
        max_steps_by_track=limits, expert_results=details, reused_reference=str(source)), indent=2))
    stage.rename(target)
    print('Prepared expert reference:', target, limits, flush=True)
    return target, limits


def merge_offline(config, source, additions, target):
    episodes = []; origins = {}
    for directory in (source, additions):
        for path in sorted(Path(directory).rglob('*.pkl.gz')):
            with gzip.open(path, 'rb') as file:
                episode = pickle.load(file)
            meta = episode['metadata']
            if meta['track'] not in config['train_tracks']:
                continue
            if meta.get('agent') not in ('team4', 'The Winners') or meta['step_size'] != config['step_size']:
                raise ValueError(f'Incompatible source episode: {path}')
            episodes.append(episode); origins[id(episode)] = path
    expected = {(t, float(e)) for t in config['train_tracks'] for e in config['offline']['epsilons']}
    actual = {(e['metadata']['track'], float(e['metadata']['epsilon'])) for e in episodes}
    if actual != expected:
        raise ValueError(f'Missing or unexpected map/epsilon groups: {expected ^ actual}')
    unique = drop_duplicates(episodes)
    report = deduplication_report(episodes, unique)
    target.mkdir(parents=True, exist_ok=True)
    retained = []
    for i, episode in enumerate(unique, 1):
        original = origins[id(episode)]
        destination = target / episode['metadata']['track'] / f'episode_{i}.pkl.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Existing episode differs: {destination}')
        else:
            shutil.copy2(original, destination)
        retained.append(dict(file=str(destination.relative_to(target)), source=str(original), sha256=digest))
    if {str(p.relative_to(target)) for p in target.rglob('*.pkl.gz')} != {r['file'] for r in retained}:
        raise ValueError('Unexpected episodes in merged offline dataset')
    write_csv(target / 'deduplication_summary.csv', report)
    (target / 'manifest.json').write_text(json.dumps(dict(retained_files=retained,
        trajectories_before=len(episodes), trajectories_after=len(unique),
        transitions=sum(len(e['actions']) for e in unique)), indent=2))
    for row in report:
        print(row, flush=True)
    print(f'Offline episodes: {len(episodes)} -> {len(unique)}; saved in {target}', flush=True)


def prepare_offline(config):
    _, limits = prepare_expert(config)
    source = ROOT / config['optimization']['reuse_offline']
    present = set()
    for path in source.rglob('*.pkl.gz'):
        with gzip.open(path, 'rb') as file:
            present.add(pickle.load(file)['metadata']['track'])
    missing = [t for t in config['train_tracks'] if t not in present]
    additions = ROOT / config['optimization']['additional_offline']
    if missing:
        c = deepcopy(config)
        c.update(train_tracks=missing, unseen_tracks=[], max_steps_by_track=limits)
        c['track_timeouts']['mode'] = 'fixed'
        c['offline']['dataset'] = str(additions)
        c['offline']['raw_dataset'] = str(additions) + '_raw'
        collect_dataset(c)
    merge_offline(config, source, additions, ROOT / config['offline']['dataset'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--profile', default='multimap_dagger')
    parser.add_argument('--stage', choices=['expert', 'offline'], required=True)
    args = parser.parse_args()
    config = OmegaConf.to_container(load_config(args.config, args.profile), resolve=True)
    (prepare_expert if args.stage == 'expert' else prepare_offline)(config)
