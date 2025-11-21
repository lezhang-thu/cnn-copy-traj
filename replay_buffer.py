import numpy as np
import random

import torch


class ReplayBuffer(object):

    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        # debug
        self.k = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action):
        data = (obs_t, action)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    @torch.no_grad()
    def _encode_sample(self, idxes):
        obses_t, actions = [], []
        for k in idxes:
            data = self._storage[k]
            obs_t, action = data

            obses_t.append(obs_t)
            actions.append(action)

        batch = dict(obs=np.asarray(obses_t), )
        x = {
            k: torch.as_tensor(v, dtype=torch.float32).cuda()
            for k, v in batch.items()
        }
        x['act'] = torch.as_tensor(np.asarray(actions), dtype=torch.int64)
        return x

    def sample(self, batch_size):
        start = self.k == 0
        if self.k + batch_size - 1 < len(self._storage):
            idxes = [self.k + _ for _ in range(batch_size)]
            self.k = (self.k + batch_size) % len(self._storage)
        else:
            idxes = [self.k + _ for _ in range(len(self._storage) - self.k)]
            self.k = 0
        #idxes = [
        #    random.randint(0,
        #                   len(self._storage) - 1) for _ in range(batch_size)
        #]
        x = self._encode_sample(idxes)
        x['start'] = start
        return x
