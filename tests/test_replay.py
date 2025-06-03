import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import pytest
import warnings
from torch.utils.data import Dataset, DataLoader

# Import directly from the package
from preqtorch import Replay, ReplayStreams, ReplayBuffer, ReplayingDataLoader

# Define a simple dataset for testing
class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = [(i, i * 2) for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

# Collate function for testing
def simple_collate_fn(batch):
    inputs, targets = zip(*batch)
    return torch.tensor(inputs), torch.tensor(targets)

def test_replay_streams():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create dataset
    dataset = SimpleDataset(size=100)

    # Create ReplayStreams
    batch_size = 10
    n_streams = 2
    replay_streams = ReplayStreams(
        dataset=dataset,
        batch_size=batch_size,
        n_streams=n_streams,
        collate_fn=simple_collate_fn
    )

    # Test update method
    indices = list(range(10))
    replay_streams.update(indices)

    # Test sample method
    sampled_batches = replay_streams.sample()

    # Verify the results
    assert len(sampled_batches) == n_streams, f"Expected {n_streams} batches, got {len(sampled_batches)}"
    for indices, batch in sampled_batches:
        assert len(indices) == batch_size, f"Expected batch size {batch_size}, got {len(indices)}"
        inputs, targets = batch
        assert inputs.shape[0] == batch_size, f"Expected inputs batch size {batch_size}, got {inputs.shape[0]}"
        assert targets.shape[0] == batch_size, f"Expected targets batch size {batch_size}, got {targets.shape[0]}"

    print("ReplayStreams test passed!")

def test_replay_buffer():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create dataset
    dataset = SimpleDataset(size=100)

    # Create ReplayBuffer
    batch_size = 10
    n_samples = 3
    replay_buffer = ReplayBuffer(
        dataset=dataset,
        batch_size=batch_size,
        n_samples=n_samples,
        collate_fn=simple_collate_fn
    )

    # Test update method
    indices = list(range(20))
    replay_buffer.update(indices)

    # Test sample method
    sampled_batches = replay_buffer.sample()

    # Verify the results
    assert len(sampled_batches) == n_samples, f"Expected {n_samples} batches, got {len(sampled_batches)}"
    for indices, batch in sampled_batches:
        assert len(indices) == batch_size, f"Expected batch size {batch_size}, got {len(indices)}"
        inputs, targets = batch
        assert inputs.shape[0] == batch_size, f"Expected inputs batch size {batch_size}, got {inputs.shape[0]}"
        assert targets.shape[0] == batch_size, f"Expected targets batch size {batch_size}, got {targets.shape[0]}"

    print("ReplayBuffer test passed!")

def test_replaying_data_loader():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create dataset
    dataset = SimpleDataset(size=100)

    # Create ReplayBuffer
    batch_size = 10
    n_samples = 3
    replay_buffer = ReplayBuffer(
        dataset=dataset,
        batch_size=batch_size,
        n_samples=n_samples,
        collate_fn=simple_collate_fn
    )

    # Create ReplayingDataLoader
    data_loader = ReplayingDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        replay=replay_buffer,
        collate_fn=simple_collate_fn
    )

    # Test iteration with expected warnings
    # We expect warnings about drawing batches without calling sample_replay()
    # These warnings are part of the expected behavior being tested

    # Create a single iterator to ensure samples_since_replay increases
    data_iter = iter(data_loader)

    # First batch - no warning yet
    batch = next(data_iter)
    inputs, targets = batch
    assert inputs.shape[0] == batch_size, f"Expected inputs batch size {batch_size}, got {inputs.shape[0]}"
    assert targets.shape[0] == batch_size, f"Expected targets batch size {batch_size}, got {targets.shape[0]}"

    # Second batch - should warn about 1 batch
    with pytest.warns(UserWarning, match="You have drawn 1 batches without calling sample_replay()."):
        batch = next(data_iter)
        inputs, targets = batch
        assert inputs.shape[0] == batch_size, f"Expected inputs batch size {batch_size}, got {inputs.shape[0]}"
        assert targets.shape[0] == batch_size, f"Expected targets batch size {batch_size}, got {targets.shape[0]}"

    # Third batch - should warn about 2 batches
    with pytest.warns(UserWarning, match="You have drawn 2 batches without calling sample_replay()."):
        batch = next(data_iter)
        inputs, targets = batch
        assert inputs.shape[0] == batch_size, f"Expected inputs batch size {batch_size}, got {inputs.shape[0]}"
        assert targets.shape[0] == batch_size, f"Expected targets batch size {batch_size}, got {targets.shape[0]}"

    # Fourth batch - should warn about 3 batches
    with pytest.warns(UserWarning, match="You have drawn 3 batches without calling sample_replay()."):
        batch = next(data_iter)
        inputs, targets = batch
        assert inputs.shape[0] == batch_size, f"Expected inputs batch size {batch_size}, got {inputs.shape[0]}"
        assert targets.shape[0] == batch_size, f"Expected targets batch size {batch_size}, got {targets.shape[0]}"

    # Test replay sampling
    sampled_batches = data_loader.sample_replay()

    # Verify the results
    assert len(sampled_batches) == n_samples, f"Expected {n_samples} batches, got {len(sampled_batches)}"
    for indices, batch in sampled_batches:
        assert len(indices) == batch_size, f"Expected batch size {batch_size}, got {len(indices)}"
        inputs, targets = batch
        assert inputs.shape[0] == batch_size, f"Expected inputs batch size {batch_size}, got {inputs.shape[0]}"
        assert targets.shape[0] == batch_size, f"Expected targets batch size {batch_size}, got {targets.shape[0]}"

    print("ReplayingDataLoader test passed!")

def main():
    print("Testing replay objects...")
    test_replay_streams()
    test_replay_buffer()
    test_replaying_data_loader()
    print("All replay tests passed!")

if __name__ == "__main__":
    main()
