"""RLPD agent: SAC + symmetric offline/online sampling, LayerNorm critic ensemble"""
import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

BUTTONS = ('brake', 'drift', 'fire', 'nitro', 'rescue')

def action_to_vector(action, learned):
    """action dict -> float32 vector in [-1, 1], one entry per learned action"""
    values = []
    for key in learned:
        value = float(np.asarray(action[key]).item())
        if key == 'acceleration':
            value = 2 * value - 1                  # [0, 1] -> [-1, 1]
        elif key != 'steer':
            value = 1.0 if value > 0 else -1.0     # button: on = 1, off = -1
        values.append(value)                       # steer is already in [-1, 1]
    return np.array(values, dtype=np.float32)


def vector_to_action(vector, learned):
    """vector in [-1, 1] -> env action dict; actions not learned stay 0 / False"""
    action = {'acceleration': 0.0, 'steer': 0.0, **{key: False for key in BUTTONS}}
    for key, value in zip(learned, np.clip(vector, -1, 1)):
        if key == 'acceleration':
            action[key] = float((value + 1) / 2)   # [-1, 1] -> [0, 1]
        elif key == 'steer':
            action[key] = float(value)
        else:
            action[key] = bool(value > 0)          # pressed if > 0
    return action




def mlp(input_size, hidden_sizes, output_size, layer_norm=False):
    """Linear -> (LayerNorm) -> ReLU blocks, then a linear output layer."""
    layers = []
    for size in hidden_sizes:
        layers.append(nn.Linear(input_size, size))
        if layer_norm:
            layers.append(nn.LayerNorm(size))      # RLPD: keeps critic values bounded
        layers.append(nn.ReLU())
        input_size = size
    return nn.Sequential(*layers, nn.Linear(input_size, output_size))


class Actor(nn.Module):
    """tanh-squashed Gaussian policy: actions always in [-1, 1]."""

    def __init__(self, obs_size, action_size, hidden_sizes):
        super().__init__()
        self.net = mlp(obs_size, hidden_sizes, 2 * action_size)  # mean + log_std per action

    def forward(self, obs, deterministic=False):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        if deterministic:                          # evaluation: no noise
            return torch.tanh(mean), None
        std = log_std.clamp(-5, 2).exp()
        noise = mean + std * torch.randn_like(mean)  # reparameterization trick
        action = torch.tanh(noise)
        # Gaussian log-prob, corrected for the tanh squashing
        log_prob = torch.distributions.Normal(mean, std).log_prob(noise)
        # Evaluate log(1 - tanh(u)^2) from u: tanh can round to +/-1 and
        # the epsilon-based correction then loses the entropy gradient entirely.
        log_prob = log_prob - 2 * (math.log(2) - noise - F.softplus(-2 * noise))
        return action, log_prob.sum(-1)


class CriticEnsemble(nn.Module):
    """num_critics independent Q(obs, action) networks, each with LayerNorm."""

    def __init__(self, obs_size, action_size, hidden_sizes, num_critics):
        super().__init__()
        self.critics = nn.ModuleList(
            mlp(obs_size + action_size, hidden_sizes, 1, layer_norm=True) for _ in range(num_critics))

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([critic(x).squeeze(-1) for critic in self.critics])  # (critics, batch)


class RLPDAgent:
    def __init__(self, config, encoder, normalizer, seed):
        cfg = config['rlpd']
        torch.manual_seed(seed)
        self.cfg, self.encoder, self.normalizer = cfg, encoder, normalizer
        self.learned = list(config['learned_actions'])
        device = cfg['device']
        if device in ('auto', 'mps'):
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        print(f'RLPD device: {self.device}', flush=True)

        obs_size, action_size = encoder.output_size, len(self.learned)
        self.actor = Actor(obs_size, action_size, cfg['hidden_sizes']).to(self.device)
        self.critic = CriticEnsemble(obs_size, action_size, cfg['hidden_sizes'], cfg['num_critics']).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)  # slow-moving copy
        self.log_alpha = torch.tensor(math.log(cfg['init_temperature']), device=self.device, requires_grad=True)
        self.target_entropy = -action_size / 2     # RLPD default

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg['temperature_lr'])
        self.name = 'RLPD agent'

    def act(self, encoded_obs, deterministic=False):
        """encoded obs -> action vector; random sample while training, mean when evaluating."""
        with torch.no_grad():
            obs = torch.as_tensor(encoded_obs, device=self.device).unsqueeze(0)
            action, _ = self.actor(obs, deterministic)
        return action[0].cpu().numpy()

    def choose_action(self, observation):
        """raw observation -> action dict; same interface as the DAgger actor."""
        encoded = self.normalizer.transform(self.encoder.encode(observation))
        return vector_to_action(self.act(encoded, deterministic=True), self.learned)

    def update_critic(self, batch):
        obs, action, reward, next_obs, terminated = batch
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            # pessimistic target: min over a random subset of the target critics
            subset = torch.randperm(self.cfg['num_critics'])[:self.cfg['num_min_critics']].tolist()
            # Only the randomly selected target critics contribute to the minimum.
            x = torch.cat([next_obs, next_action], dim=-1)
            next_q = torch.stack([self.target_critic.critics[i](x).squeeze(-1)
                                  for i in subset]).min(dim=0).values
            next_q = next_q - self.log_alpha.exp() * next_log_prob   # entropy bonus
            target = reward + self.cfg['gamma'] * (1 - terminated) * next_q  # no future after finish
        q = self.critic(obs, action)
        loss = F.mse_loss(q, target.expand_as(q))  # every critic regresses to the same target
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        with torch.no_grad():                      # target critics follow slowly
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.lerp_(param, self.cfg['tau'])
        return loss.detach()

    def update_actor(self, obs):
        action, log_prob = self.actor(obs)
        # Keep dQ/da for the actor, without computing unused critic parameter gradients.
        self.critic.requires_grad_(False)
        try:
            q = self.critic(obs, action).mean(dim=0)
            alpha = self.log_alpha.exp().detach()
            actor_loss = (alpha * log_prob - q).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        finally:
            self.critic.requires_grad_(True)
        alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        return actor_loss.detach()

    def update(self, minibatches, defer_logs=False):
        """Same UTD critic steps, then one actor/temperature step; defer scalar reads."""
        critic_loss = torch.stack([self.update_critic(batch) for batch in minibatches]).mean()
        actor_loss = self.update_actor(minibatches[-1][0])
        logs = dict(critic_loss=critic_loss, actor_loss=actor_loss,
                    temperature=self.log_alpha.exp().detach())
        return logs if defer_logs else materialize_logs(logs)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(), 'learned': self.learned, 'cfg': self.cfg,
            'normalizer_mean': torch.from_numpy(self.normalizer.mean.copy()),
            'normalizer_scale': torch.from_numpy(self.normalizer.scale.copy()),
        }, path)

    def load(self, path):
        # actor only enough to evaluate but can't resume training from this
        state = torch.load(path, map_location='cpu')  # normalizer is float64, mps can't hold it
        self.actor.load_state_dict(state['actor'])

def materialize_logs(logs):
    """One device-to-CPU transfer per episode rather than per gradient step."""
    keys = list(logs)
    if not all(isinstance(logs[key], torch.Tensor) for key in keys):
        return logs
    values = torch.stack([logs[key] for key in keys]).detach().cpu().tolist()
    return dict(zip(keys, values))
