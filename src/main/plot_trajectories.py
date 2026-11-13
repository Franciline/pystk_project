"""Record or replot matched single-map and multimap trajectories."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.transforms import Affine2D
from matplotlib.legend_handler import HandlerBase
import numpy as np
import pandas as pd
import hashlib
import shutil
from agents.dagger_actor import ActorPolicy
from agents.team4.agent4 import Agent4
from main.evaluate import comparison_load_rlpd as load_rlpd
from main.DAgger import to_action
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec
ROOT = Path(__file__).resolve().parents[2]
MODELS = ('Expert (team4)', 'DAgger', 'RLPD')
FILES = ('expert', 'dagger', 'rlpd')

class StartFlagHandler(HandlerBase):

    def create_artists(self, legend, original, xdescent, ydescent, width, height, fontsize, transform):
        return [Rectangle((-xdescent + column * width / 8, -ydescent + row * height / 2), width / 8, height / 2, facecolor='white' if (row + column) % 2 else 'black', edgecolor='none', alpha=0.5, transform=transform) for row in range(2) for column in range(8)]

def snapshot(env, step, reward, action):
    kart = env.world.karts[env.kart_ix]
    position = np.asarray(kart.location)
    velocity = np.asarray(kart.velocity)
    return dict(step=step, time_s=float(env.world.time), x=float(position[0]), y=float(position[1]), z=float(position[2]), vx=float(velocity[0]), vy=float(velocity[1]), vz=float(velocity[2]), speed_m_s=float(kart.speed), progress_m=float(kart.overall_distance), is_on_road=bool(kart.is_on_road), reward=reward, **{f'action_{k}': v for k, v in action.items()})

def collect(env, policy, seed, max_steps, destination, expert=False):
    random.seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    if expert:
        policy = Agent4(env, path_lookahead=3)
    elif hasattr(policy, 'reset_episode'):
        policy.reset_episode()
    items = []
    for item in env.world.items:
        location = np.asarray(item.location)
        items.append(dict(item_id=int(item.id), item_type=int(item.type.value), x=float(location[0]), y=float(location[1]), z=float(location[2])))
    pd.DataFrame(items, columns=['item_id', 'item_type', 'x', 'y', 'z']).to_csv(destination.with_name(destination.stem + '_items.csv'), index=False)
    rows = [snapshot(env, 0, 0.0, {})]
    for step in range(1, max_steps + 1):
        action = to_action(policy.choose_action(obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        rows.append(snapshot(env, step, float(reward), action))
        truncated = bool(truncated or (step == max_steps and (not terminated)))
        if terminated or truncated:
            break
    table = pd.DataFrame(rows)
    table.to_csv(destination, index=False)
    path = np.diff(table[['x', 'y', 'z']].to_numpy(), axis=0)
    return dict(n_steps=step, terminated=bool(terminated), truncated=truncated, traveled_distance_m=float(np.linalg.norm(path, axis=1).sum()), off_road_steps=int((~table['is_on_road'].iloc[1:]).sum()))

def widen_positions(points, nodes, width_scale):
    """Scale lateral offsets for display; use 3D proximity to distinguish bridges."""
    points = np.asarray(points, dtype=float)
    starts = nodes[:, 0]
    directions = nodes[:, 1] - starts
    fractions = np.clip(np.einsum('nsi,si->ns', points[:, None] - starts, directions) / np.maximum((directions ** 2).sum(axis=1), 1e-08), 0, 1)
    closest = starts + fractions[:, :, None] * directions
    selected = ((points[:, None] - closest) ** 2).sum(axis=2).argmin(axis=1)
    tangent = directions[selected][:, [0, 2]]
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-08)
    xy = points[:, [0, 2]]
    offsets = xy - closest[np.arange(len(points)), selected][:, [0, 2]]
    lateral = (offsets * normals).sum(axis=1)
    return xy + (width_scale - 1) * lateral[:, None] * normals

def road_offsets(nodes, widths, width_scale):
    """Join road edges at shared 3D endpoints, including the lap seam."""
    directions = nodes[:, 1][:, [0, 2]] - nodes[:, 0][:, [0, 2]]
    normals = np.column_stack([-directions[:, 1], directions[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-08)
    half_widths = widths * width_scale / 2
    offsets = np.repeat((normals * half_widths[:, None])[:, None], 2, axis=1)
    for i, end in enumerate(nodes[:, 1]):
        for j in np.flatnonzero(np.linalg.norm(nodes[:, 0] - end, axis=1) < 1e-05):
            if i == j:
                continue
            bisector = normals[i] + normals[j]
            bisector /= max(np.linalg.norm(bisector), 1e-08)
            denominator = np.dot(bisector, normals[i])
            join = bisector * ((half_widths[i] + half_widths[j]) / 2) / max(denominator, 0.5)
            offsets[i, 1] = offsets[j, 0] = join
    return offsets

def plot_trajectories(directory, output, width_scale=1.5):
    if not np.isfinite(width_scale) or width_scale <= 0:
        raise ValueError('width_scale must be positive and finite')
    directory, output = (Path(directory), Path(output))
    config = json.loads((directory / 'config.json').read_text())
    summary = pd.read_csv(directory / 'summary.csv')
    with np.load(directory / 'track.npz') as track:
        nodes, widths = (track['nodes'].astype(float), track['widths'].reshape(-1))
    segments = nodes[:, :, [0, 2]]
    offsets = road_offsets(nodes, widths, width_scale)
    polygons = np.stack([segments[:, 0] + offsets[:, 0], segments[:, 1] + offsets[:, 1], segments[:, 1] - offsets[:, 1], segments[:, 0] - offsets[:, 0]], axis=1)
    plt.style.use('default')
    plt.rcParams.update({'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor('#82AA63')
    low, high = (segments.reshape(-1, 2).min(axis=0), segments.reshape(-1, 2).max(axis=0))
    for z in np.arange(low[1] - 30, high[1] + 30, 24):
        ax.add_patch(Rectangle((low[0] - 30, z), high[0] - low[0] + 60, 12, facecolor='#769F58', edgecolor='none', alpha=0.4, zorder=0))
    ax.add_collection(PolyCollection(polygons, facecolors='#555961', edgecolors='none', zorder=1))
    for sign in (-1, 1):
        edge = segments + sign * offsets
        ax.add_collection(LineCollection(edge, colors='#EEECE0', linewidths=1.2, zorder=2))
    ax.add_collection(LineCollection(segments, colors='#D0CDC0', linestyles='dashed', linewidths=0.65, alpha=0.65, zorder=2))
    item_file = directory / 'expert_items.csv'
    if item_file.exists():
        items = pd.read_csv(item_file)
        items[['x', 'z']] = widen_positions(items[['x', 'y', 'z']].to_numpy(), nodes, width_scale)
        capsule = FancyBboxPatch((-0.3, -0.6), 0.6, 1.2, boxstyle='round,pad=0,rounding_size=.3').get_path()
        styles = {0: ('Bonus box', 's', '#2678E3'), 1: ('Banana', '^', '#DDAA00'), 2: ('Large nitro', capsule, '#009988'), 3: ('Small nitro', capsule, '#44AA99'), 4: ('Bubblegum', 'X', '#AA4499'), 6: ('Easter egg', 'o', '#882255')}
        for item_type, group in items.groupby('item_type'):
            label, marker, color = styles.get(item_type, (f'Item {item_type}', 'o', '#777777'))
            ax.scatter(group['x'], group['z'], marker=marker, color=color, s=90 if item_type == 2 else 40 if item_type == 3 else 45, edgecolors='white', linewidths=0.5, zorder=5, label=label)
    policies = list(zip(MODELS, FILES, ['#F3DC62', '#2D9CFF', '#FF8040'], ['--', '-', ':']))
    if (directory / 'bc.csv').exists():
        policies.insert(1, ('Behavioral Cloning', 'bc', '#C76BE8', '-.'))
    for name, filename, color, style in policies:
        rows = pd.read_csv(directory / f'{filename}.csv')
        xy = rows[['x', 'z']].to_numpy(dtype=float)
        jumps = np.flatnonzero(np.linalg.norm(np.diff(xy, axis=0), axis=1) > 20) + 1
        rows[['x', 'z']] = widen_positions(rows[['x', 'y', 'z']].to_numpy(), nodes, width_scale)
        xy = np.insert(rows[['x', 'z']].to_numpy(), jumps, np.nan, axis=0)
        metric = summary[summary['model'] == name].iloc[0]
        finished = str(metric['terminated']).lower() == 'true'
        ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=style, linewidth=2.2, alpha=0.8, label=f"{name}: {int(metric['n_steps'])} steps" + ('' if finished else ' (incomplete)'))
        sample_indices = sorted({int((len(rows) - 1) * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)})
        sampled = rows.iloc[sample_indices]
        for _, point in sampled.iterrows():
            vx, vz = (point.get('vx', 0), point.get('vz', 0))
            angle = np.degrees(np.arctan2(vz, vx))
            transform = Affine2D().scale(1.35).rotate_deg(angle).translate(point['x'], point['z']) + ax.transData
            for wheel_x in (-1.5, 1.2):
                for wheel_z in (-1.65, 1.0):
                    ax.add_patch(Rectangle((wheel_x, wheel_z), 1.1, 0.65, facecolor='#151515', edgecolor='none', transform=transform, zorder=7))
            ax.add_patch(Rectangle((-2.1, -1.1), 4.2, 2.2, facecolor=color, edgecolor='white', linewidth=0.55, transform=transform, zorder=8))
            ax.add_patch(Rectangle((0.5, -0.65), 1.2, 1.3, facecolor='#263541', edgecolor='none', transform=transform, zorder=9))
    start = pd.read_csv(directory / 'expert.csv').iloc[0]
    start_world = np.array([start.x, start.y, start.z])
    directions = nodes[:, 1] - nodes[:, 0]
    fractions = np.clip(((start_world - nodes[:, 0]) * directions).sum(axis=1) / np.maximum((directions ** 2).sum(axis=1), 1e-08), 0, 1)
    projected = nodes[:, 0] + fractions[:, None] * directions
    nearest = np.linalg.norm(projected - start_world, axis=1).argmin()
    center = projected[nearest][[0, 2]]
    edge_offset = (1 - fractions[nearest]) * offsets[nearest, 0] + fractions[nearest] * offsets[nearest, 1]
    span = 2 * np.linalg.norm(edge_offset)
    tile_length, thickness = (span / 10, 1.2)
    angle = np.degrees(np.arctan2(edge_offset[1], edge_offset[0]))
    flag_transform = Affine2D().rotate_deg(angle).translate(*center) + ax.transData
    for row in range(2):
        for column in range(10):
            ax.add_patch(Rectangle((-span / 2 + column * tile_length, -thickness / 2 + row * thickness / 2), tile_length, thickness / 2, facecolor='white' if (row + column) % 2 else 'black', edgecolor='none', alpha=0.5, transform=flag_transform, zorder=6))
    start_handle = Rectangle((0, 0), 1, 1, label='Start')
    ax.add_patch(start_handle)
    start_handle.set_visible(False)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylim(-275, 100)
    policy_title = 'Expert, Behavioral Cloning, DAgger and RLPD' if (directory / 'bc.csv').exists() else 'Expert, DAgger and RLPD'
    ax.set(title=config.get('plot_title', f"{policy_title} trajectories on {config['track'].capitalize()} map, seed {config['seed']}"), xlabel='X (m)', ylabel='Z (m)')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc='best', frameon=False, fontsize=10, handler_map={start_handle: StartFlagHandler()})
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / config.get('plot_stem', f"trajectories_{config['track']}_seed{config['seed']}")
    for extension in ('svg',):
        fig.savefig(stem.with_suffix('.' + extension), bbox_inches='tight')
        print(stem.with_suffix('.' + extension))
    plt.close(fig)

def run(args):
    if args.add_bc:
        config = json.loads((args.data_dir / 'config.json').read_text())
        destination = args.data_dir / 'bc.csv'
        if destination.exists():
            raise FileExistsError(f'{destination} already exists; use --plot-only to reuse it')
        policy = ActorPolicy.load(args.bc_checkpoint, device=args.device)
        env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=config['track'], num_kart=1, laps=1, step_size=config['step_size'])
        try:
            metrics = collect(env, policy, config['seed'], config['max_steps'], destination)
        finally:
            env.close()
        summary = pd.read_csv(args.data_dir / 'summary.csv')
        summary = pd.concat([summary, pd.DataFrame([dict(model='Behavioral Cloning', **metrics)])], ignore_index=True)
        summary.to_csv(args.data_dir / 'summary.csv', index=False)
        config['bc'] = str(args.bc_checkpoint.resolve())
        (args.data_dir / 'config.json').write_text(json.dumps(config, indent=2))
        print('Behavioral Cloning:', metrics, flush=True)
        plot_trajectories(args.data_dir, args.plot_dir, args.width_scale)
        return
    if args.plot_only:
        plot_trajectories(args.data_dir, args.plot_dir, args.width_scale)
        return
    dagger = ActorPolicy.load(args.dagger, device=args.device)
    rlpd = load_rlpd(args.rlpd, args.device)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    config = dict(track=args.track, seed=args.seed, max_steps=args.max_steps, step_size=args.step_size, dagger=str(args.dagger.resolve()), rlpd=str(args.rlpd.resolve()), rlpd_normalization=True)
    with (args.data_dir / 'config.json').open('x') as file:
        json.dump(config, file, indent=2)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=args.track, num_kart=1, laps=1, step_size=args.step_size)
    try:
        for name, filename, policy in zip(MODELS, FILES, [None, dagger, rlpd]):
            print(f'Recording {name}', flush=True)
            metrics = collect(env, policy, args.seed, args.max_steps, args.data_dir / f'{filename}.csv', expert=policy is None)
            if filename == 'expert':
                np.savez_compressed(args.data_dir / 'track.npz', nodes=np.asarray(env.track.path_nodes), widths=np.asarray(env.track.path_width))
            with (args.data_dir / 'summary.csv').open('a') as file:
                pd.DataFrame([dict(model=name, **metrics)]).to_csv(file, index=False, header=filename == 'expert')
            print(metrics, flush=True)
    finally:
        env.close()
    plot_trajectories(args.data_dir, args.plot_dir, args.width_scale)
multi_ROOT = Path(__file__).resolve().parents[2]
multi_POLICIES = [('Behavioral Cloning', 'bc'), ('DAgger', 'dagger'), ('RLPD', 'rlpd')]

def multi_selection(root):
    chosen = {}
    for method, filename in multi_POLICIES:
        candidates = []
        for seed in (1000, 2000, 3000):
            directory = root / f"{('rlpd' if method == 'RLPD' else 'dagger')}_seed{seed}"
            if method == 'Behavioral Cloning':
                validation = pd.read_csv(directory / 'evaluation.csv')
                row = validation[validation.iteration == 1].iloc[0].to_dict()
                checkpoint = directory / 'actor_iteration_1.pt'
            else:
                row = json.loads((directory / 'best_evaluation.json').read_text())
                checkpoint = directory / ('actor_best.pt' if method == 'DAgger' else 'agent_best.pt')
            finish = row['finish_steps_mean']
            score = (-float(row['completion_rate']), float(finish) if finish is not None and pd.notna(finish) else float('inf'), seed)
            candidates.append((score, dict(method=method, training_seed=seed, checkpoint=str(checkpoint.resolve()), validation=row)))
        _, best = min(candidates, key=lambda item: item[0])
        best['sha256'] = hashlib.sha256(Path(best['checkpoint']).read_bytes()).hexdigest()
        chosen[filename] = best
    return chosen

def multi_metrics(directory, output):
    config = json.loads((directory / 'config.json').read_text())
    summary = pd.read_csv(directory / 'summary.csv').set_index('model')
    single = Path(config['expert_reference'])
    data = {}
    for regime, source in [('Multimap', directory), ('Single map', single)]:
        for label, name in [('Expert (team4)', 'expert'), *multi_POLICIES]:
            d = pd.read_csv(source / f'{name}.csv')
            d = d[d.progress_m >= 0].copy()
            if not np.isfinite(d[['progress_m', 'x', 'z']].to_numpy()).all():
                raise ValueError('Non-finite trajectory')
            if (np.diff(d.progress_m) < 0).any():
                raise ValueError('Backtracking needs a separate alignment rule')
            d = d.drop_duplicates('progress_m', keep='first')
            data[regime, name] = d
    start = max((d.progress_m.iloc[0] for d in data.values()))
    end = min((d.progress_m.iloc[-1] for d in data.values()))
    if not end > start:
        raise ValueError('No common track-progress range')
    grid = np.linspace(start, end, int(np.ceil((end - start) / 0.1)) + 1)

    def aligned(d):
        return np.column_stack([np.interp(grid, d.progress_m, d[axis]) for axis in ('x', 'z')])
    expert = aligned(data['Multimap', 'expert'])
    rows = []
    profiles = dict(progress_m=grid)
    for regime, source in [('Multimap', directory), ('Single map', single)]:
        details = pd.read_csv(source / 'summary.csv').set_index('model')
        for label, name in [('Expert (team4)', 'expert'), *multi_POLICIES]:
            error = np.linalg.norm(aligned(data[regime, name]) - expert, axis=1)
            row = details.loc[label]
            if str(row.terminated).lower() != 'true':
                raise ValueError('Incomplete trajectory; distance needs coverage reporting')
            mean = float(np.trapezoid(error, grid) / (end - start))
            p95 = float(np.percentile(error, 95))
            if name == 'expert':
                assert mean == p95 == 0
            rows.append(dict(regime=regime, model=label, mean_distance_to_expert_m=mean, p95_distance_to_expert_m=p95, lap_steps=int(row.n_steps)))
            profiles[f"{regime.lower().replace(' ', '_')}_{name}_distance_m"] = error
    table = pd.DataFrame(rows)
    stem = output / config['plot_stem']
    table[table.regime == 'Multimap'].to_csv(stem.with_name(stem.name + '_metrics.csv'), index=False)
    table.to_csv(stem.with_name(stem.name + '_comparison.csv'), index=False)
    pd.DataFrame(profiles).to_csv(stem.with_name(stem.name + '_distance_profile.csv'), index=False)
    stem.with_name(stem.name + '_metrics.json').write_text(json.dumps(dict(source=str(directory.resolve()), expert_reference=str(single.resolve()), alignment='Equal track progress, linear interpolation, original unscaled world X/Z', progress_start_m=float(start), progress_end_m=float(end), samples=len(grid), spacing_m=float(grid[1] - grid[0]), mean='Trapezoidal distance integral / progress span', p95='95th percentile of distance samples', note='Common overlap grid includes both single-map and multimap runs; negative startup sentinel excluded', selection=config['selection']), indent=2))
    print(table.to_string(index=False))

def multi_run(args):
    directory = args.data_dir
    directory.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        chosen = multi_selection(args.root)
        reference = json.loads((args.expert_reference / 'config.json').read_text())
        config = dict(track=reference['track'], seed=reference['seed'], step_size=reference['step_size'], max_steps=reference['max_steps'], expert_reference=str(args.expert_reference.resolve()), selection=chosen, selection_rule='Training-map validation completion then macro finishing steps; test/unseen excluded', plot_stem='multimap_trajectories_abyss_seed20000', plot_title='Expert, Behavioral Cloning, DAgger and RLPD trajectories on Abyss map, seed 20000\nPolicies trained on five maps')
        path = directory / 'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Saved selection/config differs; use a fresh data directory')
        path.write_text(json.dumps(config, indent=2))
        for filename in ('expert.csv', 'expert_items.csv', 'track.npz'):
            source = args.expert_reference / filename
            destination = directory / filename
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise ValueError(f'Expert reference differs: {filename}')
            else:
                shutil.copy2(source, destination)
        if not (directory / 'summary.csv').exists():
            old = pd.read_csv(args.expert_reference / 'summary.csv')
            old[old.model == 'Expert (team4)'].to_csv(directory / 'summary.csv', index=False)
        env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False), track=config['track'], num_kart=1, laps=1, step_size=config['step_size'])
        try:
            for method, name in multi_POLICIES:
                if (directory / f'{name}.csv').exists():
                    continue
                checkpoint = Path(chosen[name]['checkpoint'])
                policy = load_rlpd(checkpoint, args.device) if method == 'RLPD' else ActorPolicy.load(checkpoint, device=args.device)
                detail = collect(env, policy, config['seed'], config['max_steps'], directory / f'{name}.csv')
                pd.DataFrame([dict(model=method, **detail)]).to_csv(directory / 'summary.csv', mode='a', header=False, index=False)
                print(method, 'training seed', chosen[name]['training_seed'], detail, flush=True)
        finally:
            env.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectories(directory, args.output_dir, width_scale=args.width_scale)
    multi_metrics(directory, args.output_dir)

def main():
    parser = argparse.ArgumentParser(description='Record or replot matched trajectories for one setup.')
    parser.add_argument('--setup', choices=['singlemap', 'multimap'], required=True)
    parser.add_argument('--plot-only', action='store_true', help='Use saved trajectories; no simulator runs')
    parser.add_argument('--root', type=Path, default=ROOT / 'datasets/multimap/full5maps50k')
    parser.add_argument('--expert-reference', type=Path, default=ROOT / 'datasets/protocol/trajectories_abyss_seed20000_normalized')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', '--plot-dir', dest='output_dir', type=Path, default=ROOT / 'datasets/plots')
    parser.add_argument('--width-scale', type=float, default=4)
    parser.add_argument('--dagger', type=Path, default=ROOT / 'datasets/protocol/best_dagger_gaussian/actor_best.pt')
    parser.add_argument('--rlpd', type=Path, default=ROOT / 'datasets/protocol/rlpd_seed1000/agent_best.pt')
    parser.add_argument('--bc-checkpoint', type=Path, default=ROOT / 'datasets/protocol/dagger_offline_gaussian_seed1000/actor_iteration_1.pt')
    parser.add_argument('--add-bc', action='store_true')
    parser.add_argument('--track', default='abyss')
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--step-size', type=float, default=0.1)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    args = parser.parse_args()
    if args.width_scale <= 0:
        parser.error('--width-scale must be positive')
    if args.setup == 'multimap':
        if args.add_bc:
            parser.error('Multimap automatically includes BC')
        args.data_dir = args.data_dir or args.root / 'trajectories_abyss_seed20000'
        reference = json.loads((args.expert_reference / 'config.json').read_text())
        if reference['track'] != args.track or reference['seed'] != args.seed:
            parser.error('Multimap uses the saved expert lap; --track and --seed must match its config')
        multi_run(args)
    else:
        args.data_dir = args.data_dir or ROOT / 'datasets/protocol/trajectories_abyss_seed20000_normalized'
        args.plot_dir = args.output_dir
        run(args)
if __name__ == '__main__':
    main()
