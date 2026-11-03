"""MLP actor and supervised DAgger training on aggregated expert labels."""
import csv
import gzip
from pathlib import Path
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.observation_encoder import ObservationEncoder
from utils.observation_normalizer import ObservationNormalizer

BUTTONS = ('brake', 'drift', 'fire', 'nitro', 'rescue')


class Actor(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        for size in hidden_sizes:
            layers.extend([nn.Linear(input_size, size), nn.ReLU()])
            input_size = size
        self.network = nn.Sequential(*layers, nn.Linear(input_size, 7))

    def forward(self, inputs):
        raw = self.network(inputs)
        # Boolean outputs remain logits for BCEWithLogitsLoss.
        return torch.cat((raw[..., :1].sigmoid(), raw[..., 1:2].tanh(), raw[..., 2:]), dim=-1)


def expert_targets(action):
    keys = ('acceleration', 'steer') + BUTTONS
    return np.array([np.asarray(action[k]).item() for k in keys], dtype=np.float32)


class ActorPolicy:
    def __init__(self, config, output_dir, seed):
        if (config['epochs'] < 1 or config['batch_size'] < 1
                or config['learning_rate'] <= 0
                or any(n < 1 for n in config['hidden_sizes'])
                or min(config['continuous_weight'], config['boolean_weight']) < 0
                or config['continuous_weight'] + config['boolean_weight'] <= 0):
            raise ValueError('Invalid actor training configuration')
        
        torch.manual_seed(seed)
        self.config = dict(config)
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.encoder = ObservationEncoder(config['num_paths'], config['num_items'])
        self.normalizer = ObservationNormalizer(self.encoder)
        device = config['device']
        if device in ('auto', 'mps'):
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        print(f'Actor device: {self.device}', flush=True)
        self.model = Actor(self.encoder.output_size, config['hidden_sizes']).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.generator = torch.Generator().manual_seed(seed)

    def choose_action(self, observation):
        inputs = self.normalizer.transform(self.encoder.encode(observation))
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.from_numpy(inputs).to(self.device)).cpu().numpy()
        return {
            'acceleration': float(outputs[0]), 'steer': float(outputs[1]),
            **{key: bool(outputs[i + 2] > 0) for i, key in enumerate(BUTTONS)},
        }

    @staticmethod
    def _pairs(paths):
        # Trusted self-collected pickle files only. Load one episode at a time.
        for path in paths:
            with gzip.open(path, 'rb') as file:
                episode = pickle.load(file)
            if len(episode['observations']) != len(episode['expert_actions']) + 1:
                raise ValueError(f'Misaligned episode: {path}')
            yield from zip(episode['observations'][:-1], episode['expert_actions'])

    def train_iteration(self, paths, iteration):
        """Fit normalization only on initial data; keep weights/optimizer across rounds."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.normalizer.mean is None:
            self.normalizer.fit(obs for obs, _ in self._pairs(paths))
            self.normalizer.save(self.output_dir / 'normalizer.npz')

        inputs, targets = [], []
        for obs, action in self._pairs(paths):
            inputs.append(self.normalizer.transform(self.encoder.encode(obs)))
            targets.append(expert_targets(action))
        if not inputs:
            raise ValueError('No training examples')
        
        dataset = TensorDataset(torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets)))
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True,
                            generator=self.generator)
        
        loss_path = self.output_dir / 'losses.csv'
        new_file = not loss_path.exists()

        with loss_path.open('a', newline='') as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(['iteration', 'epoch', 'samples', 'mse', 'bce', 'loss'])

            for epoch in range(1, self.config['epochs'] + 1):
                self.model.train()
                totals = np.zeros(3)

                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    prediction = self.model(x)

                    mse = nn.functional.mse_loss(prediction[:, :2], y[:, :2])
                    bce = nn.functional.binary_cross_entropy_with_logits(prediction[:, 2:], y[:, 2:])
                    
                    loss = self.config['continuous_weight'] * mse + self.config['boolean_weight'] * bce
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    totals += np.array([mse.item(), bce.item(), loss.item()]) * len(x)

                totals /= len(dataset)
                writer.writerow([iteration, epoch, len(dataset), *totals.tolist()])
                file.flush()
                print(f'Iteration {iteration}, epoch {epoch}: MSE={totals[0]:.5f}, BCE={totals[1]:.5f}, loss={totals[2]:.5f}', flush=True)
        
        self.model.eval()
        torch.save({
            'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'config': self.config, 'iteration': iteration, 'seed': self.seed,
            'normalizer_mean': torch.from_numpy(self.normalizer.mean.copy()),
            'normalizer_scale': torch.from_numpy(self.normalizer.scale.copy()),
            'categories': self.encoder.categories,
        }, self.output_dir / f'actor_iteration_{iteration}.pt')


def train_policy(learner, paths, iteration):
    learner.train_iteration(paths, iteration)
