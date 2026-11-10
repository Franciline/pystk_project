"""Live check: PYTHONPATH=<patched-build>:src python src/unit_tests/test_rescue_binding.py."""
import pickle
from pystk2_gymnasium.envs import STKRaceEnv,AgentSpec
from utils.episode_diagnostics import EpisodeDiagnostics
from main.DAgger import to_action


if __name__=='__main__':
    env=STKRaceEnv(agent=AgentSpec(rank_start=0,use_ai=False),track='abyss',num_kart=1,laps=1,step_size=.1)
    try:
        env.reset(seed=20000);d=EpisodeDiagnostics(env)
        action=to_action(dict(acceleration=.5,steer=0,brake=0,drift=0,nitro=0,fire=0,rescue=0))
        observed=False
        for step in range(65):
            action['rescue']=step==35 # Only this binding test requests manual rescue.
            env.step(action);d.capture()
            if env.world.karts[0].rescue_active:
                observed=True
                assert pickle.loads(pickle.dumps(env.world)).karts[0].rescue_active
        assert observed and d.summary()['rescue_count']>=1,d.summary()
        print('Native rescue, subprocess transport and pickle round trip passed:',d.summary())
        env.reset(seed=20000);d=EpisodeDiagnostics(env)
        action.update(acceleration=0,brake=False,rescue=False)
        for _ in range(100):
            env.step(action);d.capture()
        assert d.summary()['stuck_count']==1 and d.summary()['stuck_duration_s']>=5,d.summary()
        print('Live stationary episode detected once after countdown:',d.summary())
    finally:
        env.close()
