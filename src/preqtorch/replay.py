import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from copy import deepcopy
import os, random
import warnings

def _make_indexed_collate_fn(base_collate_fn):
    '''
    Helper function which creates a collate function to work with the indexed dataset.
    Args:
        base_collate_fn: User supplied collate function.

    Returns:
        indexed_collate: Collate function for indexed dataset.

    '''
    def indexed_collate(batch):
        indices, data = zip(*batch)
        return list(indices), base_collate_fn(data)
    return indexed_collate


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, idx):
        return idx, self.base_dataset[idx]

    def __len__(self):
        return len(self.base_dataset)

class Replay:
    def __init__(self, dataset, collate_fn=None):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def update(self, new_indices):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class ReplayStreams(Replay):
    def __init__(self, dataset, batch_size, n_streams, reset_prob_fn=None, collate_fn=None, num_workers=0):
        super().__init__(dataset, collate_fn)
        self.batch_size = batch_size
        self.n_streams = n_streams
        self.reset_prob_fn = reset_prob_fn or (lambda t: 1 / (t + 2))
        self.num_workers = num_workers

        self.t = 0
        self.batch_records = []
        self.stream_indices = [0 for _ in range(n_streams)]

    def update(self, new_indices):
        self.batch_records.append(new_indices)

    def sample(self):
        sampled_batches = []
        for i in range(self.n_streams):
            batch_idx = self.stream_indices[i]
            sampled_idx = self.batch_records[batch_idx]
            if random.random() < self.reset_prob_fn(self.t):
                self.stream_indices[i] = 0
            else:
                self.stream_indices[i] = min(self.stream_indices[i] + 1, len(self.batch_records) - 1)

            # We already know the exact indices and batch size, so we can directly index
            # and apply the collate function without needing a DataLoader
            samples = [self.dataset[j] for j in sampled_idx]
            batch = self.collate_fn(samples)
            sampled_batches.append((sampled_idx, batch))

        self.t += 1
        return sampled_batches


class ReplayBuffer(Replay):
    def __init__(self, dataset, batch_size, n_samples, collate_fn=None, num_workers=0):
        """
        Uniformly samples from previously seen indices.

        Args:
            dataset: PyTorch dataset.
            n_samples: Number of batches to sample per call.
            collate_fn: Function to collate individual samples into a batch.
            num_workers: For optional future DataLoader use.
        """
        super().__init__(dataset, collate_fn)
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.seen_indices = []
        self.num_workers = num_workers

    def update(self, new_indices):
        self.seen_indices.extend(new_indices)

    def sample(self):
        """
        Returns `n_samples` batches, each of `batch_size`, sampled uniformly from seen indices.
        """
        sampled_batches = []
        for _ in range(self.n_samples):
            sampled_idx = random.sample(self.seen_indices, self.batch_size)
            samples = [self.dataset[j] for j in sampled_idx]
            batch = self.collate_fn(samples)
            sampled_batches.append((sampled_idx, batch))
        return sampled_batches


class ReplayingDataLoader:
    def __init__(self, dataset, batch_size, replay: Replay, shuffle=True, collate_fn=None, warn_threshold=1):
        self.indexed_dataset = IndexedDataset(dataset)
        self.collate_fn = collate_fn or torch.utils.data._utils.collate.default_collate
        self.replay = replay
        self.batch_size = batch_size
        self.warn_threshold = warn_threshold

        self.loader = DataLoader(
            self.indexed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_make_indexed_collate_fn(self.collate_fn)
        )
        self.iterator = iter(self.loader)
        self.samples_since_replay = 0
        self.warned = False

    def __iter__(self):
        return self

    def __next__(self):
        indices, batch = next(self.iterator)
        self.replay.update(indices)
        if self.samples_since_replay >= self.warn_threshold:
            warnings.warn(
                f"You have drawn {self.samples_since_replay} batches without calling sample_replay().",
                stacklevel=2
            )
        self.samples_since_replay += 1
        return batch

    def sample_replay(self):
        self.samples_since_replay = 0
        self.warned = False
        return self.replay.sample()
