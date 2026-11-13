"""Symmetric replay with identical sampling order and one transfer per field/UTD group."""
import numpy as np
import torch

FIELDS = ('obs', 'action', 'reward', 'next_obs', 'terminated')


class SymmetricSampler:
    def __init__(self, offline, online, batch_size, utd_ratio, rng, device):
        if batch_size < 2 or batch_size % 2 or utd_ratio < 1:
            raise ValueError('Require an even batch size >= 2 and a positive UTD ratio')
        self.offline, self.online = offline, online
        self.half, self.utd = batch_size // 2, utd_ratio
        self.rng, self.device = rng, device
        self.staging = {key: np.empty((utd_ratio, batch_size, *offline[key].shape[1:]), dtype=np.float32)
                        for key in FIELDS}

    def __call__(self, count):
        if count < self.half:
            raise ValueError('Insufficient online replay')
        # Preserve the original offline/online draw order for every UTD minibatch.
        for update in range(self.utd):
            off = self.rng.integers(0, len(self.offline['reward']), self.half)
            on = self.rng.integers(0, count, self.half)
            for key in FIELDS:
                np.take(self.offline[key], off, axis=0, out=self.staging[key][update, :self.half])
                np.take(self.online[key], on, axis=0, out=self.staging[key][update, self.half:])
        # Blocking transfers protect reusable CPU staging memory on MPS as well as CPU.
        tensors = [torch.from_numpy(self.staging[key]).to(self.device) for key in FIELDS]
        return [tuple(tensor[i] for tensor in tensors) for i in range(self.utd)]
