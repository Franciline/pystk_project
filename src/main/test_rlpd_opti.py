"""Check optimized replay, SAC updates, cache reuse and optional live parallel evaluation."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.rlpd_agent import RLPDAgent as Original
from agents.rlpd_agent_opti import RLPDAgent as Optimized, materialize_logs
from main.RLPD import sample, FIELDS
from utils.rlpd_replay_opti import SymmetricSampler
from utils.rlpd_dataset_opti import prepare_offline
from utils.rlpd_dataset import load_episodes, drop_duplicates, build_offline_buffer
from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer

ROOT=Path(__file__).resolve().parents[2]


def config_for_test(directory, device='cpu'):
    c=OmegaConf.to_container(OmegaConf.load(ROOT/'src/main/full5maps50k_rlpd_opti.yaml'),resolve=True)
    c.update(train_tracks=['abyss','volcano_island'],unseen_tracks=['sandtrack'],
        output_dir=str(directory/'run'), total_env_steps=9,eval_every=3,max_steps=3,
        max_steps_by_track={t:3 for t in ('abyss','volcano_island','sandtrack')})
    c['track_timeouts']['mode']='fixed'
    c.pop('expert_benchmark',None)
    c['encoder']=dict(num_paths=2,num_items=2)
    c['rlpd'].update(device=device,hidden_sizes=[8],batch_size=4,utd_ratio=10,num_critics=5,num_min_critics=2)
    c['evaluation']['seeds']=[10000,10001]
    c['optimization'].update(offline_cache=str(directory/'cache'),torch_threads=1,evaluation_workers=2)
    dataset=directory/'offline';dataset.mkdir()
    for track in c['train_tracks']:
        source=sorted((ROOT/'datasets/multimap/offline5'/track).rglob('*.pkl.gz'))[0]
        with gzip.open(source,'rb') as file:episode=pickle.load(file)
        episode['observations']=episode['observations'][:9]
        for key in ('actions','expert_actions','rewards','terminated','truncated'):
            episode[key]=episode[key][:8]
        with gzip.open(dataset/f'{track}.pkl.gz','wb') as file:pickle.dump(episode,file)
    c['offline']['dataset']=str(dataset)
    return c


def check_cache(config,directory):
    outputs=[]
    for name in ('first','second'):
        path=directory/name;path.mkdir()
        outputs.append(prepare_offline(config,path))
    encoder,normalizer,buffer,summary=outputs[0]
    episodes=drop_duplicates(load_episodes(root=Path(config['offline']['dataset']),agents=config['offline']['agents'],
        epsilons=config['offline']['epsilons'],tracks=config['train_tracks']))
    expected_normalizer=ObservationNormalizer(encoder).fit(o for e in episodes for o in e['observations'][:-1])
    expected=build_offline_buffer(episodes,encoder,expected_normalizer,config['learned_actions'])
    for key in FIELDS:
        np.testing.assert_array_equal(buffer[key],expected[key])
        np.testing.assert_array_equal(outputs[1][2][key],expected[key])
        assert not buffer[key].flags.writeable
    np.testing.assert_array_equal(normalizer.mean,expected_normalizer.mean)
    np.testing.assert_array_equal(normalizer.scale,expected_normalizer.scale)
    assert summary==outputs[1][3]
    assert len(list((directory/'cache').glob('*/manifest.json')))==1
    raw=deepcopy(config);raw['normalize_observations']=False
    path=directory/'raw';path.mkdir()
    _, identity, _, _=prepare_offline(raw,path)
    np.testing.assert_array_equal(identity.mean,np.zeros(encoder.output_size))
    np.testing.assert_array_equal(identity.scale,np.ones(encoder.output_size))
    assert len(list((directory/'cache').glob('*/manifest.json')))==2
    print('PASS: replay cache matches original arrays/statistics, is reused read-only, separates normalization modes')
    return encoder,normalizer,buffer


def check_updates(config,encoder,normalizer,offline):
    torch.set_num_threads(1)
    original=Original(config,encoder,normalizer,1000)
    optimized=Optimized(config,encoder,normalizer,1000)
    online={key:np.array(value[:7],copy=True) for key,value in offline.items()}
    rng1,rng2=np.random.default_rng(1000),np.random.default_rng(1000)
    sampler=SymmetricSampler(offline,online,4,10,rng2,optimized.device)
    for step in range(3):
        reference=[]
        for _ in range(10):
            off=sample(offline,len(offline['reward']),2,rng1)
            on=sample(online,7,2,rng1)
            reference.append(tuple(torch.as_tensor(np.concatenate([a,b])).to(original.device) for a,b in zip(off,on)))
        batches=sampler(7)
        for a,b in zip(reference,batches):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        before=torch.get_rng_state()
        device_before=torch.mps.get_rng_state() if original.device.type=='mps' else None
        old_logs=original.update(reference)
        after=torch.get_rng_state()
        device_after=torch.mps.get_rng_state() if original.device.type=='mps' else None
        torch.set_rng_state(before)
        if device_before is not None:torch.mps.set_rng_state(device_before)
        logs=materialize_logs(optimized.update(batches,defer_logs=True))
        assert torch.equal(after,torch.get_rng_state())
        if device_after is not None:assert torch.equal(device_after,torch.mps.get_rng_state())
        tolerance=0 if original.device.type=='cpu' else 2e-6
        for name in ('actor','critic','target_critic'):
            for x,y in zip(getattr(original,name).parameters(),getattr(optimized,name).parameters()):
                torch.testing.assert_close(x,y,rtol=tolerance,atol=tolerance)
        torch.testing.assert_close(original.log_alpha,optimized.log_alpha,rtol=tolerance,atol=tolerance)
        np.testing.assert_allclose(list(old_logs.values()),list(logs.values()),rtol=2e-6,atol=2e-6)
        for name in ('actor_opt','critic_opt','alpha_opt'):
            old=getattr(original,name).state_dict()['state']
            new=getattr(optimized,name).state_dict()['state']
            for index in old:
                for key in old[index]:
                    torch.testing.assert_close(old[index][key],new[index][key],rtol=tolerance,atol=tolerance)
        assert all(p.requires_grad for p in optimized.critic.parameters())
    assert rng1.bit_generator.state==rng2.bit_generator.state
    print(f'PASS: identical replay draws/RNG streams and matching SAC parameters across 3 updates on {optimized.device}')
    return optimized


def check_live(config,directory,agent):
    from pystk2_gymnasium.envs import STKRaceEnv,AgentSpec
    from main.DAgger import to_action
    from utils.multitrack import evaluate_tracks as sequential
    from utils import multitrack_rlpd_opti as parallel
    from utils.rlpd_evaluation_opti import load_policy
    env=STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track='abyss',num_kart=1,laps=1,step_size=.1)
    checkpoint=directory/'agent.pt';agent.save(checkpoint)
    cpu_policy=load_policy(checkpoint,config)
    cpu_policy.evaluation_checkpoint=checkpoint
    c=deepcopy(config);c['max_steps']=60;c['max_steps_by_track']={}
    try:
        expected=sequential(env,cpu_policy,c,directory/'sequential',to_action)
        parallel.start_workers(c)
        actual=parallel.evaluate_tracks(env,cpu_policy,c,directory/'parallel',to_action)
        assert actual==expected
        again=parallel.evaluate_tracks(env,cpu_policy,c,directory/'parallel_again',to_action)
        assert again==expected
        for first in (directory/'sequential').rglob('*.pkl.gz'):
            second=directory/'parallel'/first.relative_to(directory/'sequential')
            with gzip.open(first,'rb') as file:a=pickle.load(file)
            with gzip.open(second,'rb') as file:b=pickle.load(file)
            assert a['actions']==b['actions']
            assert a['diagnostic_summary']==b['diagnostic_summary']
            for x,y in zip(a['observations'],b['observations']):
                for key in x:
                    if isinstance(x[key],str):assert x[key]==y[key]
                    else:np.testing.assert_allclose(x[key],y[key],rtol=1e-6,atol=1e-6)
        print('PASS: sequential/parallel RLPD evaluations, records and diagnostics match; worker reuse')
    finally:
        parallel.stop_workers();env.close()
    from main.RLPD_opti import run_rlpd
    run_rlpd(config)
    output=Path(config['output_dir'])
    import csv
    with (output/'evaluation_by_track.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==9 and {r['split'] for r in rows}=={'train','unseen'}
    with (output/'online_episodes.csv').open() as f:episodes=list(csv.DictReader(f))
    assert len(episodes)==3 and [int(r['env_steps']) for r in episodes]==[3,6,9]
    assert (output/'timings.csv').exists() and (output/'agent_best.pt').exists() and (output/'agent_last.pt').exists()
    for path in (output/'online_trajectories').rglob('*.pkl.gz'):
        with gzip.open(path,'rb') as f:episode=pickle.load(f)
        assert episode['expert_actions']==[None]*len(episode['actions'])
        assert all(not action['rescue'] for action in episode['actions'])
        assert len(episode['observations'])==len(episode['actions'])+1
    loaded=load_policy(output/'agent_best.pt',config)
    assert np.isfinite(next(loaded.actor.parameters()).detach().numpy()).all()
    print('PASS: end-to-end RLPD runner, updates, 3 evaluations, timings, checkpoint loading and records')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args()
    if args.device=='mps' and not torch.backends.mps.is_available():parser.error('MPS is unavailable')
    with TemporaryDirectory(prefix='rlpd-opti-') as name:
        directory=Path(name)
        config=config_for_test(directory,args.device)
        encoder,normalizer,offline=check_cache(config,directory)
        agent=check_updates(config,encoder,normalizer,offline)
        if args.live:check_live(config,directory,agent)
