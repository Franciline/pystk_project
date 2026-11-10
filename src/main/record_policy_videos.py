"""Record solo actor laps and overlay their actual next N positions afterward."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main.plot_trajectory_screenshot import project
from main.compare_trajectories import ROOT
from main.compare_generalization import load_rlpd
from main.DAgger import to_action
from agents.dagger_actor import ActorPolicy
from pystk2_gymnasium.envs import STKRaceEnv, AgentSpec


def overlay(frame, positions, views, projections, index, horizon, label):
    image = frame.copy()
    draw = ImageDraw.Draw(image)
    points = positions[index:min(index+horizon+1, len(positions))].copy()
    points[:, 1] += .2
    ndc, valid = project(points, views[index], projections[index])
    pixels = np.column_stack([(ndc[:,0]+1)*image.width/2, (1-ndc[:,1])*image.height/2])
    if valid[0]:
        x,y = pixels[0]
        draw.line((x,y-45,x,y+12), fill='white', width=3)
    for x,y in pixels[1:][valid[1:]]:
        draw.ellipse((x-5,y-5,x+5,y+5),fill='#FF8040',outline='white',width=1)
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    draw.rectangle((image.width*.23,8,image.width*.77,68),fill=(20,25,30))
    draw.text((image.width/2,13),label,anchor='mt',fill='white',font=font)
    draw.text((image.width/2,39),f'Actual next {horizon} steps - recorded rollout',
              anchor='mt',fill='#FFB080',font=font)
    return image


def record(args, model, ffmpeg):
    checkpoint = args.rlpd if model == 'rlpd' else args.dagger
    policy = load_rlpd(checkpoint,args.device) if model == 'rlpd' else ActorPolicy.load(checkpoint,device=args.device)
    training_config = json.loads((checkpoint.parent/'config.json').read_text())
    training_seed = training_config.get('seed')
    name = 'RLPD' if model == 'rlpd' else 'DAgger'
    if model == 'dagger' and training_config.get('exploration',{}).get('mode') == 'gaussian':
        name += ' Gaussian'
    label = name + f' - seed {training_seed}'
    stem = f'{model}_{args.track}_seed{args.seed}'
    run_dir = args.record_dir / stem
    run_dir.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed)
    env = STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track=args.track,
                     num_kart=1,laps=1,step_size=args.step_size,render_mode='human',
                     use_subprocess=False,num_cameras=1)
    positions,views,projections,times,actions = [],[],[],[],[]
    with TemporaryDirectory(prefix='stk-video-') as scratch:
        scratch = Path(scratch)
        try:
            obs,_ = env.reset(seed=args.seed)
            for step in range(args.max_steps+1):
                camera = env.world.players[0].camera
                if camera is None:
                    raise RuntimeError('Camera matrices unavailable. Use the patched PySTK binding; see patches/README.md.')
                positions.append(np.asarray(env.world.karts[env.kart_ix].location).copy())
                views.append(np.asarray(camera.view).copy())
                projections.append(np.asarray(camera.projection).copy())
                times.append(float(env.world.time))
                frame = Image.fromarray(np.asarray(env._stk.race.screen_capture()).copy())
                frame = frame.resize((960, int(round(frame.height*960/frame.width/2))*2),Image.Resampling.LANCZOS)
                frame.save(scratch/f'{step:05d}.png',compress_level=1)
                if step == args.max_steps or (step > 0 and (terminated or truncated)):
                    break
                action = to_action(policy.choose_action(obs))
                actions.append(action)
                obs,_,terminated,truncated,_ = env.step(action)
                if (step+1)%100 == 0:
                    print(f'{model}: {step+1} steps recorded',flush=True)
        finally:
            env.close()
        positions,views,projections = map(np.asarray,(positions,views,projections))
        truncated = bool(truncated or (step == args.max_steps and not terminated))
        preview_start = min(args.preview_start,max(0,len(positions)*args.step_size-args.preview_seconds))
        np.savez_compressed(run_dir/'states.npz',positions=positions,views=views,
                            projections=projections,times=times)
        metadata = dict(model=model,training_seed=training_seed,checkpoint=str(checkpoint.resolve()),track=args.track,
                        seed=args.seed,step_size=args.step_size,future_steps=args.future_steps,
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        n_steps=len(positions)-1,terminated=bool(terminated),truncated=bool(truncated),
                        dots='Actual future positions from this same graphical rollout; not direct network position predictions',
                        normalization=True,preview_start_seconds=preview_start,
                        preview_seconds=args.preview_seconds,actions=actions)
        (run_dir/'recording.json').write_text(json.dumps(metadata,indent=2))
        for index in range(len(positions)):
            path = scratch/f'{index:05d}.png'
            with Image.open(path) as raw:
                rendered = overlay(raw.convert('RGB'),positions,views,projections,index,args.future_steps,label)
            rendered.save(path,compress_level=1)
            if index == min(385,len(positions)-1):
                rendered.save(args.output_dir/f'{stem}_poster.png')
        mp4 = args.output_dir/f'{stem}.mp4'
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-framerate',str(1/args.step_size),
                        '-i',str(scratch/'%05d.png'),'-c:v','libx264','-crf','21','-pix_fmt','yuv420p',
                        '-movflags','+faststart',str(mp4)],check=True)
        filters = 'fps=5,scale=400:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3'
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(mp4),
                        '-ss',str(preview_start),'-t',str(args.preview_seconds),
                        '-filter_complex',filters,'-loop','0',str(args.output_dir/f'{stem}.gif')],check=True)
        print(f'{model}: finished={terminated}, steps={len(positions)-1}; saved {mp4}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=['rlpd','dagger'],default=['rlpd','dagger'])
    parser.add_argument('--rlpd',type=Path,default=ROOT/'datasets/protocol/rlpd_seed1000/agent_best.pt')
    parser.add_argument('--dagger',type=Path,default=ROOT/'datasets/protocol/best_dagger_gaussian/actor_best.pt')
    parser.add_argument('--track',default='abyss')
    parser.add_argument('--seed',type=int,default=20000)
    parser.add_argument('--future-steps',type=int,default=10)
    parser.add_argument('--step-size',type=float,default=.1)
    parser.add_argument('--preview-start',type=float,default=30,help='GIF preview start in seconds')
    parser.add_argument('--preview-seconds',type=float,default=16,help='GIF duration; MP4 keeps the full lap')
    parser.add_argument('--max-steps',type=int,default=1000)
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'assets/rollouts')
    parser.add_argument('--record-dir',type=Path,default=ROOT/'datasets/protocol/rollout_videos')
    args=parser.parse_args()
    if args.preview_start<0 or args.preview_seconds<=0:
        parser.error('preview-start must be nonnegative and preview-seconds must be positive')
    if args.future_steps<1 or args.step_size<=0 or args.max_steps<1:
        parser.error('future-steps, step-size and max-steps must be positive')
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:
        parser.error('FFmpeg is required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for model in args.models:
        record(args,model,ffmpeg)
