"""Run: PYTHONPATH=src python src/unit_tests/test_rlpd_action_numerics.py."""
import math
import numpy as np
import torch
from agents.rlpd_agent import Actor,action_to_vector,vector_to_action


def test_numerics():
    learned=['acceleration','steer','brake','drift','fire','nitro']
    original=dict(acceleration=.8,steer=-.3,brake=False,drift=True,fire=False,nitro=True,rescue=False)
    mapped=vector_to_action(action_to_vector(original,learned),learned)
    assert abs(mapped['acceleration']-.8)<1e-6 and abs(mapped['steer']+.3)<1e-6
    assert all(mapped[key]==original[key] for key in learned[2:])
    # Extreme actor means must retain a gradient pushing them away from saturation.
    actor=Actor(2,2,[])
    with torch.no_grad():
        actor.net[-1].weight.zero_()
        actor.net[-1].bias.copy_(torch.tensor([-100.,100.,0.,0.]))
    torch.manual_seed(123)
    actions,log_prob=actor(torch.zeros(1,2))
    assert torch.equal(actions,torch.tensor([[-1.,1.]]))
    assert torch.isfinite(log_prob).all()
    log_prob.sum().backward()
    gradient=actor.net[-1].bias.grad
    assert torch.isfinite(gradient).all()
    assert torch.allclose(gradient[:2],torch.tensor([-2.,2.]),atol=1e-5)
    # Match the usual Jacobian formula where tanh is safely away from +/-1.
    with torch.no_grad():actor.net[-1].bias.copy_(torch.tensor([.1,-.2,-1.,-1.]))
    torch.manual_seed(44)
    u=torch.tensor([[.1,-.2]])+math.exp(-1)*torch.randn(1,2)
    expected=(torch.distributions.Normal(torch.tensor([[.1,-.2]]),math.exp(-1)).log_prob(u)
              -torch.log(1-torch.tanh(u).square())).sum(-1)
    torch.manual_seed(44)
    actions,actual=actor(torch.zeros(1,2))
    assert torch.allclose(actual,expected,atol=1e-6)
    assert torch.allclose(actions,torch.tanh(u))
    # Evaluation still uses exactly tanh(mean), without noise.
    deterministic,_=actor(torch.zeros(1,2),deterministic=True)
    assert torch.allclose(deterministic,torch.tanh(torch.tensor([[.1,-.2]])))


if __name__=='__main__':
    test_numerics();print('RLPD action conversion and stable tanh-gradient checks passed')
