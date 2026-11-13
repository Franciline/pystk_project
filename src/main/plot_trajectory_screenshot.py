"""Overlay recorded world trajectories on a real STK camera image."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main.compare_trajectories import ROOT
from main.compare_generalization import load_rlpd
from agents.dagger_actor import ActorPolicy
from main.DAgger import to_action
from agents.team4.agent4 import Agent4
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec


def project(points, view, projection):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    # Irrlicht stores column-major matrices, exposed as C-order 4x4 memoryviews.
    clip = homogeneous @ view @ projection
    valid = clip[:, 3] > 0
    ndc = clip[:, :3] / np.where(valid, clip[:, 3], 1)[:, None]
    valid &= (np.abs(ndc[:, :2]) <= 1).all(axis=1)
    valid &= (np.abs(ndc[:, 2]) <= 1)
    return ndc, valid


def run(args):
    if args.step < 0 or args.future_steps < 1:
        raise ValueError('step must be nonnegative and future-steps must be positive')
    config = json.loads((args.data_dir / 'config.json').read_text())
    policy = None
    if args.model == 'rlpd':
        policy = load_rlpd(Path(config['rlpd']), args.device)
    elif args.model == 'dagger':
        policy = ActorPolicy.load(Path(config['dagger']), device=args.device)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0, use_ai=False),
                     track=config['track'], num_kart=1, laps=1,
                     step_size=config['step_size'], render_mode='human', use_subprocess=False, num_cameras=1)
    try:
        obs, _ = env.reset(seed=config['seed'])
        if policy is None:
            policy = Agent4(env, path_lookahead=3)
        elif hasattr(policy, 'reset_episode'):
            policy.reset_episode()
        for _ in range(args.step):
            obs, _, terminated, truncated, _ = env.step(to_action(policy.choose_action(obs)))
            if terminated or truncated:
                raise ValueError('The episode ended before the requested screenshot')
        frame = np.asarray(env._stk.race.screen_capture()).copy()
        current = np.asarray(env.world.karts[env.kart_ix].location).copy()
        camera = env.world.players[0].camera
        view = np.asarray(camera.view).copy() if camera is not None else None
        projection = np.asarray(camera.projection).copy() if camera is not None else None
        current_time = float(env.world.time)
        future_positions = []
        for ahead in range(1, args.future_steps+1):
            obs, _, terminated, truncated, _ = env.step(to_action(policy.choose_action(obs)))
            future_positions.append(np.asarray(env.world.karts[env.kart_ix].location).copy())
            if terminated or truncated:
                break
        future_positions = np.asarray(future_positions)
        future = np.asarray(env.world.karts[env.kart_ix].location).copy()
        future_time = float(env.world.time)
    finally:
        env.close()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame)
    height, width = frame.shape[:2]
    metadata = dict(model=args.model, track=config['track'], seed=config['seed'],
                    step=args.step, future_steps=ahead, elapsed_seconds=future_time-current_time,
                    current_world=current.tolist(), future_world=future.tolist(),
                    future_positions=future_positions.tolist(),
                    future_pixels=[None]*len(future_positions),
                    camera_available=view is not None, current_pixel=None, future_pixel=None)
    if view is not None:
        points = np.vstack([current, future_positions])
        points[:, 1] += .2
        ndc, valid = project(points, view, projection)
        pixels = np.column_stack([(ndc[:,0]+1)*width/2, (1-ndc[:,1])*height/2])
        if valid[0]:
            x, y = pixels[0]
            metadata['current_pixel'] = [float(x),float(y)]
        for index in np.flatnonzero(valid[1:]):
            metadata['future_pixels'][index] = pixels[index+1].tolist()
        visible_pixels = pixels[1:][valid[1:]]
        ax.scatter(visible_pixels[:,0],visible_pixels[:,1],s=45,
                   color='#FF8040',edgecolors='white',linewidths=1)
        for index in sorted({0,len(future_positions)-1}):
            if valid[index+1]:
                x,y = pixels[index+1]
                ax.annotate(f'+{index+1}', (x,y), xytext=(10,12),
                            textcoords='offset points',color='white')
        metadata['future_pixel'] = metadata['future_pixels'][-1]
    else:
        ax.text(.02,.96,'Camera matrices unavailable - future projection unavailable',
                transform=ax.transAxes,color='white',va='top',
                bbox=dict(facecolor='black',alpha=.55,edgecolor='none'))
        print('Camera matrices unavailable; no future-position dot was drawn.')
    ax.set(xlim=(0,width),ylim=(height,0))
    ax.axis('off')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f'trajectory_screenshot_{args.model}_{config["track"]}_step{args.step}'
    for suffix in ('svg',):
        fig.savefig(stem.with_suffix('.'+suffix),bbox_inches='tight',pad_inches=0)
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    if view is not None:
        np.savez_compressed(stem.with_suffix('.npz'), frame=frame, view=view,
                            projection=projection, current=current, future=future,
                            future_positions=future_positions)
    plt.close(fig)
    print(stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'datasets/protocol/trajectories_abyss_seed20000_normalized')
    parser.add_argument('--step', type=int, default=150)
    parser.add_argument('--model', choices=['expert','dagger','rlpd'],default='rlpd')
    parser.add_argument('--future-steps',type=int,default=10,
                        help='Show every future position from +1 through +N steps')
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'datasets/plots')
    run(parser.parse_args())
