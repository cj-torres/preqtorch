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


def test_replay_implementations_use_default_collation():
    dataset = SimpleDataset(size=4)
    replay_implementations = [
        ReplayBuffer(dataset, batch_size=2, n_samples=1),
        ReplayStreams(dataset, batch_size=2, n_streams=1),
    ]

    for replay in replay_implementations:
        replay.update([0, 1])
        sampled_batches = replay.sample()

        assert len(sampled_batches) == 1
        indices, (inputs, targets) = sampled_batches[0]
        assert sorted(indices) == [0, 1]
        assert sorted(inputs.tolist()) == [0, 1]
        assert torch.equal(targets, inputs * 2)


def test_replay_buffer_caps_batch_size_at_seen_population():
    dataset = SimpleDataset(size=4)
    replay = ReplayBuffer(dataset, batch_size=4, n_samples=1)
    replay.update([2])

    sampled_batches = replay.sample()

    assert len(sampled_batches) == 1
    indices, (inputs, targets) = sampled_batches[0]
    assert indices == [2]
    assert torch.equal(inputs, torch.tensor([2]))
    assert torch.equal(targets, torch.tensor([4]))


def test_replay_streams_returns_no_samples_before_first_update():
    replay = ReplayStreams(SimpleDataset(size=4), batch_size=2, n_streams=1)

    assert replay.sample() == []


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda dataset: ReplayBuffer(dataset, batch_size=0, n_samples=1), "positive integer batch_size"),
        (lambda dataset: ReplayBuffer(dataset, batch_size=2, n_samples=-1), "non-negative integer"),
        (lambda dataset: ReplayStreams(dataset, batch_size=2, n_streams=-1), "non-negative integer"),
    ],
)
def test_replay_implementations_reject_invalid_sizes(factory, message):
    with pytest.raises(ValueError, match=message):
        factory(SimpleDataset(size=4))


def test_replaying_data_loader_rejects_replay_for_different_dataset():
    dataset = SimpleDataset(size=4)
    other_dataset = SimpleDataset(size=4)
    replay = ReplayBuffer(other_dataset, batch_size=2, n_samples=1)

    with pytest.raises(ValueError, match="must use the ReplayingDataLoader dataset"):
        ReplayingDataLoader(dataset, batch_size=2, replay=replay)


def test_replaying_data_loader_requires_batch_size_without_source_loader():
    dataset = SimpleDataset(size=4)
    replay = ReplayStreams(dataset, batch_size=None, n_streams=1)

    with pytest.raises(ValueError, match="positive integer without a source dataloader"):
        ReplayingDataLoader(dataset, batch_size=None, replay=replay)


def test_replaying_data_loader_is_reiterable_and_reports_batch_count():
    dataset = SimpleDataset(size=5)
    replay = ReplayBuffer(dataset, batch_size=2, n_samples=0)
    data_loader = ReplayingDataLoader(
        dataset=dataset,
        batch_size=2,
        replay=replay,
        shuffle=False,
        warn_threshold=100,
    )

    first_pass = [inputs.tolist() for inputs, _ in data_loader]
    second_pass = [inputs.tolist() for inputs, _ in data_loader]

    assert len(data_loader) == 3
    assert first_pass == [[0, 1], [2, 3], [4]]
    assert second_pass == first_pass

def main():
    print("Testing replay objects...")
    test_replay_streams()
    test_replay_buffer()
    test_replaying_data_loader()
    print("All replay tests passed!")

if __name__ == "__main__":
    main()
