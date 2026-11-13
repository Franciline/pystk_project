"""Tiny live plumbing check, not a learning experiment. Requires patched PySTK."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import gzip,pickle,json,csv
import numpy as np
import torch
from omegaconf import OmegaConf
from agents.dagger_actor import ActorPolicy,train_policy
from main.DAgger import collect_episode,run_dagger
from main.RLPD import run_rlpd
from pystk2_gymnasium.envs import STKRaceEnv,AgentSpec

ROOT=Path(__file__).resolve().parents[2]


from utils.experiment_config import load_config

def smoke():
    with TemporaryDirectory(prefix='stk-multimap-check-') as directory:
        root=Path(directory);data=root/'offline'
        env=STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track='abyss',num_kart=1,laps=1,step_size=.1)
        try:
            for track in ['abyss','volcano_island','sandtrack']:
                print('Smoke collecting',track,flush=True)
                env.default_track=track
                collect_episode(env,None,beta=1,seed=1,max_steps=5,
                    output=data/track/'team4.pkl.gz',iteration=0,step_size=.1)
        finally:env.close()
        common=dict(train_tracks=['abyss','volcano_island','sandtrack'],unseen_tracks=['snowtuxpeak','xr591','zengarden'],
            max_steps=5,max_steps_by_track=dict(abyss=5,volcano_island=6,sandtrack=7,
                snowtuxpeak=8,xr591=9,zengarden=10),expert_benchmark=None,seed=1000,online_seed_start=1000,
            evaluation=dict(seeds=[10000],min_steps_improvement=0),
            diagnostics=dict(enabled=True,save_evaluation_episodes=True,stuck_seconds=5,stuck_radius_m=1))
        c=OmegaConf.to_container(load_config(profile='singlemap_dagger'),resolve=True)
        c.update(deepcopy(common));c.update(N=3,steps_per_iteration=15,output_dir=str(root/'dagger'))
        c['initialization'].update(mode='load_offline',dataset=str(data))
        c['actor'].update(num_paths=182,num_items=10,hidden_sizes=[8],batch_size=4,epochs=1,device='cpu')
        policy=ActorPolicy(c['actor'],root/'dagger',c['seed'])
        run_dagger(c,learner=policy,train_policy=train_policy)
        c=OmegaConf.to_container(load_config(profile='singlemap_rlpd'),resolve=True)
        c.update(deepcopy(common));c.update(total_env_steps=36,eval_every=18,output_dir=str(root/'rlpd'))
        c['offline']['dataset']=str(data);c['encoder'].update(num_paths=182,num_items=10)
        c['rlpd'].update(hidden_sizes=[8],batch_size=4,utd_ratio=1,num_critics=2,num_min_critics=2,device='cpu')
        run_rlpd(c)
        for method in ['dagger','rlpd']:
            folder=root/method
            assert (folder/'evaluation_by_track.csv').exists()
            files=list((folder/'evaluation_episodes').rglob('*.pkl.gz'))
            assert all(any(track in str(p) for p in files) for track in common['unseen_tracks'])
            for p in files:
                with gzip.open(p,'rb') as file:episode=pickle.load(file)
                assert len(episode['diagnostics'])==len(episode['actions'])+1
                assert episode['diagnostic_summary']['rescue_status_available']
                assert len(episode['actions'])==common['max_steps_by_track'][episode['metadata']['track']]
            log='metrics.csv' if method=='dagger' else 'online_episodes.csv'
            with (folder/log).open() as f:rows=list(csv.DictReader(f))
            for track in common['train_tracks']:
                assert [int(r['seed']) for r in rows if r['track']==track]==[1000,1001]
            with (folder/'collection_by_track.csv').open() as f:exposure=list(csv.DictReader(f))
            assert [int(r['cumulative_trajectories']) for r in exposure[-3:]]==[2,2,2]
        dagger_state=torch.load(root/'dagger/actor_best.pt',map_location='cpu',weights_only=True)
        assert torch.any(dagger_state['normalizer_mean']!=0)
        state=torch.load(root/'rlpd/agent_best.pt',map_location='cpu',weights_only=True)
        assert torch.all(state['normalizer_mean']==0) and torch.all(state['normalizer_scale']==1)
        with gzip.open(next((root/'rlpd/online_trajectories').glob('*.pkl.gz')),'rb') as file:
            episode=pickle.load(file)
        assert all(a is None for a in episode['expert_actions'])
        assert all(not a['rescue'] for a in episode['actions'])
        print('Both cores trained, evaluated train/unseen maps, and saved compatible diagnostics; RLPD inputs unnormalized')


if __name__=='__main__':smoke()
