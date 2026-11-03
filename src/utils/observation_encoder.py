"""Fixed-size solo-policy inputs. Raw observations remain untouched.

Layout: numeric fields, global categories, path rows, item rows.
Path row: start(3), end(3), distances(2), width(1), valid(1).
Item row: position(3), one-hot type, valid(1).
Each path contributes 10 inputs: 3 + 3 + 2 + 1 + 1.
Each item contributes 4 + number of item categories: 3 + categories + 1.
With 6 item categories, 182 paths and 10 items contribute 1820 + 100 inputs;
numeric kart fields and global one-hot categories supply the remaining inputs.
Normalization is deliberately separate; these outputs retain physical units.
"""
import numpy as np
import pystk2
from typing import ClassVar


class ObservationEncoder:
    NUMERIC_FIELDS: ClassVar[dict[str, int]] = { # number elements per fields
        'velocity': 3, 'front': 3, 'max_steer_angle': 1, 'jumping': 1,
        'center_path': 3, 'center_path_distance': 1, 'distance_down_track': 1,
        'powerup_count': 1, 'energy': 1, 'shield_time': 1,
        'attachment_time_left': 1, 'aux_ticks': 1, 'skeed_factor': 1,
    }

    def __init__(self, num_paths=182, num_items=10):
        if type(num_paths) is not int or num_paths < 1:
            raise ValueError('num_paths must be a positive integer')
        if type(num_items) is not int or num_items < 0:
            raise ValueError('num_items must be a nonnegative integer')
        self.num_paths = num_paths
        self.num_items = num_items
        self.categories = { # encoding categories
            'powerup_type': self._ids(pystk2.Powerup),
            'attachment': self._ids(pystk2.Attachment),
            'phase': [0, 1, 2, 3],  # Local envs.Phase, not pystk2.WorldState.Phase.
            'items_type': self._ids(pystk2.Item),
        }
        self.output_size = (
            sum(self.NUMERIC_FIELDS.values())
            + sum(len(self.categories[k]) for k in ('powerup_type', 'attachment', 'phase'))
            + num_paths * 10 + num_items * (4 + len(self.categories['items_type']))
        )

    @staticmethod
    def _ids(enum_cls):
        return sorted({int(v.value) for v in enum_cls.Type.__members__.values()})

    def normalization_mask(self, vector):
        """Select continuous/count features, excluding padded slots and jumping."""
        vector = np.asarray(vector)
        if vector.shape != (self.output_size,):
            raise ValueError('Encoded observation has the wrong shape')
        mask = np.zeros(self.output_size, dtype=bool)
        offset = 0
        for field, size in self.NUMERIC_FIELDS.items():
            mask[offset:offset + size] = field != 'jumping'
            offset += size
        offset += sum(len(self.categories[k]) for k in ('powerup_type', 'attachment', 'phase'))
        for _ in range(self.num_paths):
            mask[offset:offset + 9] = vector[offset + 9] == 1
            offset += 10
        width = 4 + len(self.categories['items_type'])
        for _ in range(self.num_items):
            mask[offset:offset + 3] = vector[offset + width - 1] == 1
            offset += width
        return mask

    def _one_hot(self, field, value):
        ids = self.categories[field]
        value = np.asarray(value).item()
        if value not in ids:
            raise ValueError(f'Unknown {field}: {value}')
        result = np.zeros(len(ids), dtype=np.float32)
        result[ids.index(value)] = 1
        return result

    def encode(self, obs):
        """Return a fresh float32 vector; reject non-solo observations.

        kart_type is ignored. AI action labels are never used as inputs.
        """
        if len(obs['karts_position']):
            raise ValueError('Encoder supports solo observations only')
        parts = []
        for field, size in self.NUMERIC_FIELDS.items():
            value = np.asarray(obs[field], dtype=np.float32).reshape(-1)
            if value.size != size:
                raise ValueError(f'{field}: expected {size} values, got {value.size}')
            parts.append(value)
        for field in ('powerup_type', 'attachment', 'phase'):
            parts.append(self._one_hot(field, obs[field]))

        fields = [('paths_start', 3), ('paths_end', 3),
                  ('paths_distance', 2), ('paths_width', 1)]
        lengths = [len(obs[key]) for key, _ in fields]
        if len(set(lengths)) != 1:
            raise ValueError('Path fields must have matching lengths')
        paths = np.zeros((self.num_paths, 10), dtype=np.float32)
        count = min(lengths[0], self.num_paths)
        offset = 0
        for key, width in fields:
            values = np.asarray(obs[key], dtype=np.float32).reshape(lengths[0], width)
            paths[:count, offset:offset + width] = values[:count]
            offset += width
        paths[:count, -1] = 1
        parts.append(paths.ravel())

        if len(obs['items_position']) != len(obs['items_type']):
            raise ValueError('Item positions and types must have matching lengths')
        items = np.zeros((self.num_items, 4 + len(self.categories['items_type'])), dtype=np.float32)
        count = min(len(obs['items_type']), self.num_items)
        # Missing type slots are conceptually -1: encoded as zero vector + mask=0.
        for i in range(count):
            items[i, :3] = obs['items_position'][i]
            items[i, 3:-1] = self._one_hot('items_type', obs['items_type'][i])
            items[i, -1] = 1
        parts.append(items.ravel())
        result = np.concatenate(parts).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError('Observation contains non-finite values')
        return result
