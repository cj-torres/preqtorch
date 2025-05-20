import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import os, random

class ReplayStreams:
    def __init__(self, n_streams, reset_prob_fn=None):
        """
        Manages replay streams for online learning.

        Args:
            n_streams: Number of independent replay streams.
            reset_prob_fn: Function f(t) giving probability of reset at iteration t (default = 1/(t+2))
        """
        self.dataset = []
        self.n_streams = n_streams
        self.stream_indices = [0 for _ in range(n_streams)]
        self.reset_prob_fn = reset_prob_fn or (lambda t: 1 / (t + 2))
        self.t = 0

    def update(self, batch):
        self.dataset.append(batch)

    def sample(self):
        """
        Returns n_streams batches for the given iteration t.
        May reset stream index with probability defined by reset_prob_fn.

        Returns:
            List[Tuple[batch_idx, batch_data]]
        """
        sampled = []
        for i in range(self.n_streams):
            index = self.stream_indices[i]
            batch = self.dataset[index]
            sampled.append((index, batch))

            if random.random() < self.reset_prob_fn(self.t):
                self.stream_indices[i] = 0
            else:
                self.stream_indices[i] = min(self.stream_indices[i] + 1, len(self.dataset) - 1)

        self.t += 1

        return sampled


class ReplayBuffer:
    def __init__(self, n_samples):
        """
        Manages replay streams for online learning.

        Args:
            n_streams: Number of independent replay streams.
            reset_prob_fn: Function f(t) giving probability of reset at iteration t (default = 1/(t+2))
        """
        self.dataset = []
        self.n_samples = n_samples

    def update(self, batch):
        self.dataset.append(batch)

    def sample(self):
        """
        Returns n_streams batches for the given iteration t.
        May reset stream index with probability defined by reset_prob_fn.

        Returns:
            List[Tuple[batch_idx, batch_data]]
        """

        return random.choices(self.dataset, k=self.n_samples)


