import random
import warnings
from numbers import Integral

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset, default_collate


def _materialize_batch(dataset, sampled_idx, collate_fn):
    """Materialize a replay batch, using fast-path tensor indexing when available."""
    if collate_fn is default_collate and isinstance(dataset, TensorDataset):
        index_tensor = torch.as_tensor(sampled_idx, dtype=torch.long)
        return tuple(tensor.index_select(0, index_tensor.to(tensor.device)) for tensor in dataset.tensors)

    samples = [dataset[j] for j in sampled_idx]
    return collate_fn(samples)

class _IndexedCollate:
    """Apply a source DataLoader's collate function while retaining sample indices."""

    def __init__(self, base_collate_fn):
        self.base_collate_fn = base_collate_fn

    def __call__(self, batch):
        indices, data = zip(*batch)
        return list(indices), self.base_collate_fn(data)


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
        if collate_fn is None:
            collate_fn = default_collate
        self.collate_fn = collate_fn

    def update(self, new_indices):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class ReplayStreams(Replay):
    def __init__(self, dataset, batch_size, n_streams, reset_prob_fn=None, collate_fn=None, num_workers=0):
        super().__init__(dataset, collate_fn)
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        if isinstance(n_streams, bool) or not isinstance(n_streams, Integral) or n_streams < 0:
            raise ValueError("n_streams must be a non-negative integer")
        self.batch_size = batch_size
        self.n_streams = int(n_streams)
        self.reset_prob_fn = reset_prob_fn or (lambda t: 1 / (t + 2))
        self.num_workers = num_workers

        self.t = 0
        self.batch_records = []
        self.stream_indices = [0 for _ in range(n_streams)]

    def update(self, new_indices):
        self.batch_records.append(new_indices)

    def sample(self):
        if not self.batch_records:
            return []

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
            batch = _materialize_batch(self.dataset, sampled_idx, self.collate_fn)
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
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError("ReplayBuffer requires a positive integer batch_size")
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 0:
            raise ValueError("n_samples must be a non-negative integer")
        self.n_samples = int(n_samples)
        self.batch_size = int(batch_size)
        self.seen_indices = []
        self.num_workers = num_workers

    def update(self, new_indices):
        self.seen_indices.extend(new_indices)

    def sample(self):
        """
        Returns `n_samples` batches sampled uniformly from seen indices.

        Batches contain up to `batch_size` items when fewer indices have been seen.
        """
        sample_size = min(self.batch_size, len(self.seen_indices))
        if sample_size == 0:
            return []

        sampled_batches = []
        for _ in range(self.n_samples):
            sampled_idx = random.sample(self.seen_indices, sample_size)
            batch = _materialize_batch(self.dataset, sampled_idx, self.collate_fn)
            sampled_batches.append((sampled_idx, batch))
        return sampled_batches


class _ReplayingIterator:
    def __init__(self, replaying_loader):
        self.replaying_loader = replaying_loader
        self.iterator = iter(replaying_loader.loader)

    def __iter__(self):
        return self

    def __next__(self):
        return self.replaying_loader._next_batch(self.iterator)


class ReplayingDataLoader:
    @classmethod
    def from_dataloader(cls, dataloader, replay, collate_fn=None, warn_threshold=1, pin_memory=None):
        """Wrap an existing DataLoader without replacing its batch sampling policy."""
        if isinstance(dataloader.dataset, IterableDataset):
            raise TypeError("source dataloader must use a map-style dataset")
        if dataloader.batch_sampler is None:
            raise ValueError("source dataloader must use automatic or custom batching")
        batch_size = dataloader.batch_size
        if batch_size is None:
            batch_size = getattr(dataloader.batch_sampler, "batch_size", None)
        return cls(
            dataloader.dataset,
            batch_size=batch_size,
            replay=replay,
            collate_fn=dataloader.collate_fn if collate_fn is None else collate_fn,
            warn_threshold=warn_threshold,
            pin_memory=pin_memory,
            source_dataloader=dataloader,
        )

    def __init__(self, dataset, batch_size, replay: Replay, shuffle=True, collate_fn=None,
                 warn_threshold=1, pin_memory=None, source_dataloader=None):
        if isinstance(dataset, IterableDataset):
            raise TypeError("ReplayingDataLoader requires a map-style dataset")
        if not isinstance(replay, Replay):
            raise TypeError("replay must be an instance of Replay")
        if replay.dataset is not dataset:
            raise ValueError("replay must use the ReplayingDataLoader dataset")
        if source_dataloader is not None and source_dataloader.dataset is not dataset:
            raise ValueError("source_dataloader must use the replay dataset")
        if source_dataloader is None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer without a source dataloader")

        self.indexed_dataset = IndexedDataset(dataset)
        if collate_fn is None:
            collate_fn = default_collate
        self.collate_fn = collate_fn
        self.replay = replay
        self.batch_size = batch_size
        self.warn_threshold = warn_threshold

        indexed_collate = _IndexedCollate(self.collate_fn)
        if source_dataloader is None:
            self.loader = DataLoader(
                self.indexed_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=indexed_collate,
                pin_memory=False if pin_memory is None else pin_memory,
            )
        else:
            loader_kwargs = {
                "batch_sampler": source_dataloader.batch_sampler,
                "collate_fn": indexed_collate,
                "num_workers": source_dataloader.num_workers,
                "pin_memory": source_dataloader.pin_memory if pin_memory is None else pin_memory,
                "timeout": source_dataloader.timeout,
                "worker_init_fn": source_dataloader.worker_init_fn,
                "multiprocessing_context": source_dataloader.multiprocessing_context,
                "generator": source_dataloader.generator,
                "persistent_workers": source_dataloader.persistent_workers,
            }
            if source_dataloader.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = source_dataloader.prefetch_factor
            if hasattr(source_dataloader, "pin_memory_device"):
                loader_kwargs["pin_memory_device"] = source_dataloader.pin_memory_device
            if hasattr(source_dataloader, "in_order"):
                loader_kwargs["in_order"] = source_dataloader.in_order
            self.loader = DataLoader(self.indexed_dataset, **loader_kwargs)
        self.iterator = None
        self.samples_since_replay = 0

    def __iter__(self):
        return _ReplayingIterator(self)

    def __next__(self):
        if self.iterator is None:
            self.iterator = _ReplayingIterator(self)
        return next(self.iterator)

    def _next_batch(self, iterator):
        indices, batch = next(iterator)
        self.replay.update(indices)
        if self.samples_since_replay >= self.warn_threshold:
            warnings.warn(
                f"You have drawn {self.samples_since_replay} batches without calling sample_replay().",
                stacklevel=2
            )
        self.samples_since_replay += 1
        return batch

    def __len__(self):
        return len(self.loader)

    def sample_replay(self):
        self.samples_since_replay = 0
        return self.replay.sample()
