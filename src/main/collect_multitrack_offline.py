"""Collect one shared Team4 epsilon-mixture dataset for multi-map DAgger and RLPD."""
import argparse
from pathlib import Path
import sys
from omegaconf import OmegaConf

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main.DAgger import collect_episode
from pystk2_gymnasium.envs import STKRaceEnv,AgentSpec
from utils.multitrack import train_tracks

ROOT=Path(__file__).resolve().parents[2]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('rlpd_config.yaml'))
    parser.add_argument('--experiment',type=Path)
    args,overrides=parser.parse_known_args()
    c=OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(args.config),
        OmegaConf.load(args.experiment) if args.experiment else {},OmegaConf.from_dotlist(overrides)),resolve=True)
    tracks=train_tracks(c); offline=c['offline']; target=Path(offline['dataset'])
    if not target.is_absolute(): target=ROOT/target
    env=STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track=tracks[0],
                   num_kart=1,laps=1,step_size=c['step_size'])
    try:
        for track in tracks:
            env.default_track=track
            for epsilon in offline['epsilons']:
                for seed in range(offline['seed_start'],offline['seed_start']+offline['episodes_per_epsilon']):
                    output=target/track/f'team4_eps{float(epsilon)}_seed{seed}.pkl.gz'
                    if output.exists():
                        print(f'Already collected: {output}',flush=True)
                        continue
                    metrics=collect_episode(env,None,beta=1,seed=seed,max_steps=c['max_steps'],
                        output=output,iteration=0,step_size=c['step_size'],epsilon=epsilon,
                        diagnostics=c.get('diagnostics'))
                    print(f'{track}, epsilon={epsilon}, seed={seed}: {metrics["n_steps"]} steps',flush=True)
    finally:
        env.close()
