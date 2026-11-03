"""Fit once on training inputs, then reuse frozen statistics during inference."""
import json
from pathlib import Path
import numpy as np


class ObservationNormalizer:
    def __init__(self, encoder):
        self.encoder = encoder
        self.mean = None
        self.scale = None

    def fit(self, observations):
        """Consume raw training observations once; use per-feature population std.

        Streaming Welford updates avoid keeping the encoded dataset in RAM.
        Padding does not contribute. Constant/unseen features use scale 1.
        A fitted normalizer cannot be refitted accidentally.
        """
        if self.mean is not None:
            raise RuntimeError('Statistics are frozen; create a new normalizer to refit')
        count = np.zeros(self.encoder.output_size, dtype=np.int64)
        mean = np.zeros(self.encoder.output_size, dtype=np.float64)
        m2 = np.zeros_like(mean)
        seen = False
        for obs in observations:
            vector = self.encoder.encode(obs).astype(np.float64)
            mask = self.encoder.normalization_mask(vector)
            count[mask] += 1
            delta = vector[mask] - mean[mask]
            mean[mask] += delta / count[mask]
            m2[mask] += delta * (vector[mask] - mean[mask])
            seen = True
        if not seen:
            raise ValueError('Cannot fit on empty training data')
        variance = np.divide(m2, count, out=np.zeros_like(m2), where=count > 0)
        scale = np.sqrt(np.maximum(variance, 0))
        scale[scale < 1e-6] = 1
        self.mean, self.scale = mean, scale
        return self

    def transform(self, vector):
        """Normalize an encoded vector; preserve categories, masks and padding."""
        if self.mean is None:
            raise RuntimeError('Fit or load statistics first')
        result = np.array(vector, dtype=np.float32, copy=True)
        mask = self.encoder.normalization_mask(result)
        if not np.isfinite(result).all():
            raise ValueError('Non-finite encoded observation')
        result[mask] = (result[mask] - self.mean[mask]) / self.scale[mask]
        return result

    def _signature(self):
        return json.dumps({
            'num_paths': self.encoder.num_paths, 'num_items': self.encoder.num_items,
            'categories': self.encoder.categories,
            'numeric_fields': list(self.encoder.NUMERIC_FIELDS.items()),
            'layout_version': 1,
        }, sort_keys=True)

    def save(self, path):
        if self.mean is None:
            raise RuntimeError('Fit statistics before saving')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as file:
            np.savez_compressed(file, mean=self.mean, scale=self.scale,
                                signature=self._signature())

    @classmethod
    def load(cls, path, encoder):
        result = cls(encoder)
        with np.load(path, allow_pickle=False) as data:
            if data['signature'].item() != result._signature():
                raise ValueError('Normalizer and encoder settings do not match')
            mean, scale = data['mean'].copy(), data['scale'].copy()
        if (mean.shape != (encoder.output_size,) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise ValueError('Invalid normalization statistics')
        result.mean, result.scale = mean, scale
        return result
