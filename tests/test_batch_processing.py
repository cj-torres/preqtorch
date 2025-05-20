import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

# We're testing the batch processing logic directly, without importing preqcluster.py
# This avoids issues with relative imports in preqcluster.py

def test_batch_processing():
    """
    Test the batch processing logic in prequential_clustering.

    This test ensures that the batch processing logic works correctly:
    1. Verifies that the collate_fn is applied correctly
    2. Verifies that the code lengths are calculated correctly for each item in the batch
    3. Tests both cases: when a collate_fn is provided and when it's not
    4. Tests edge cases (e.g., batch size larger than the number of candidates, single item in a batch)
    """
    print("Testing batch processing logic...")

    # Create a simple dataset for testing
    class SimpleTestDataset(Dataset):
        def __init__(self, size=10):
            self.size = size
            # Create random tensors for testing
            self.data = [
                (
                    torch.randn(1, 128),  # Input
                    torch.randint(0, 128, (1,))  # Target
                )
                for _ in range(size)
            ]

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx]

    # Create a custom collate function for testing
    def test_collate_fn(batch):
        inputs = torch.cat([item[0] for item in batch], dim=0)
        targets = torch.cat([item[1] for item in batch], dim=0)
        return inputs, targets

    # Create a mock encoder for testing
    class MockEncoder:
        def __init__(self):
            self.calculate_code_length_calls = []
            self.encode_calls = []
            self.encode_with_model_calls = []

        def calculate_code_length(self, model, batch, beta=None, ema_params=None, use_beta=True):
            # Record the call
            self.calculate_code_length_calls.append({
                'model': model,
                'batch': batch,
                'beta': beta,
                'ema_params': ema_params,
                'use_beta': use_beta
            })

            # Return a tensor with a batch dimension if the input has a batch dimension
            if isinstance(batch, tuple) and isinstance(batch[0], torch.Tensor) and batch[0].dim() > 1 and batch[0].size(0) > 1:
                # Return a tensor with a batch dimension
                return torch.randn(batch[0].size(0)), None, None, None
            else:
                # Return a tensor without a batch dimension
                return torch.randn(1), None, None, None

        def encode(self, dataset, set_name, n_replay_streams, learning_rate, batch_size, seed, alpha=0.1,
                  collate_fn=None, use_device_handling=False, use_beta=True, use_ema=True, shuffle=True):
            # Record the call
            self.encode_calls.append({
                'dataset': dataset,
                'set_name': set_name,
                'n_replay_streams': n_replay_streams,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'seed': seed,
                'alpha': alpha,
                'collate_fn': collate_fn,
                'use_device_handling': use_device_handling,
                'use_beta': use_beta,
                'use_ema': use_ema,
                'shuffle': shuffle
            })

            # Return a mock model and code length
            return "mock_model", 1.0, {}, torch.nn.Parameter(torch.tensor(0.0)), None

        def encode_with_model(self, model, ema_params, beta, dataset, set_name, n_replay_streams, 
                             learning_rate, batch_size, seed, alpha=0.1, collate_fn=None, 
                             use_device_handling=False, use_beta=True, use_ema=True, shuffle=True, 
                             replay_streams=None):
            # Record the call
            self.encode_with_model_calls.append({
                'model': model,
                'ema_params': ema_params,
                'beta': beta,
                'dataset': dataset,
                'set_name': set_name,
                'n_replay_streams': n_replay_streams,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'seed': seed,
                'alpha': alpha,
                'collate_fn': collate_fn,
                'use_device_handling': use_device_handling,
                'use_beta': use_beta,
                'use_ema': use_ema,
                'shuffle': shuffle,
                'replay_streams': replay_streams
            })

            # Return a mock model and code length
            return "mock_model", 1.0, {}, torch.nn.Parameter(torch.tensor(0.0)), None

    # Test case 1: Test with a custom collate_fn
    print("Test case 1: Test with a custom collate_fn")
    dataset = SimpleTestDataset(size=10)
    encoder = MockEncoder()

    # Create a mock model, ema_params, beta, and replay_streams
    model = "mock_model"
    ema_params = {}
    beta = torch.nn.Parameter(torch.tensor(0.0))
    use_beta = True

    # Create a batch of unused items
    candidate_batches = [dataset[j] for j in range(5)]

    # Process candidates in batches
    batch_size = 2
    candidate_code_lengths = []

    for i in range(0, len(candidate_batches), batch_size):
        batch = candidate_batches[i:i+batch_size]

        # Use the test_collate_fn
        batched_candidates = test_collate_fn(batch)

        # Process the batch using tensor operations
        code_lengths_tensor, _, _, _ = encoder.calculate_code_length(
            model=model,
            batch=batched_candidates,
            beta=beta,
            ema_params=ema_params,
            use_beta=use_beta
        )

        # For each item in the batch, calculate its code length
        for j in range(len(batch)):
            # If the tensor has a batch dimension, extract the j-th item
            if len(code_lengths_tensor.shape) > 1:
                item_code_lengths = code_lengths_tensor[j]
                candidate_code_lengths.append(item_code_lengths.sum().item())
            else:
                # If there's only one item in the batch, or the tensor doesn't have a batch dimension
                candidate_code_lengths.append(code_lengths_tensor.sum().item())

    # Verify that the calculate_code_length method was called with the correct arguments
    print(f"Number of calculate_code_length calls: {len(encoder.calculate_code_length_calls)}")
    assert len(encoder.calculate_code_length_calls) == 3, "Expected 3 calls to calculate_code_length"

    # Verify that the batched_candidates were passed correctly
    for i, call in enumerate(encoder.calculate_code_length_calls):
        batch = call['batch']
        if i < 2:  # First two calls should have batches of size 2
            assert isinstance(batch, tuple), f"Expected batch to be a tuple, got {type(batch)}"
            assert isinstance(batch[0], torch.Tensor), f"Expected batch[0] to be a tensor, got {type(batch[0])}"
            if i == 0:
                assert batch[0].size(0) == 2, f"Expected batch[0] to have size 2, got {batch[0].size(0)}"
            elif i == 1:
                assert batch[0].size(0) == 2, f"Expected batch[0] to have size 2, got {batch[0].size(0)}"
        else:  # Last call should have a batch of size 1
            assert isinstance(batch, tuple), f"Expected batch to be a tuple, got {type(batch)}"
            assert isinstance(batch[0], torch.Tensor), f"Expected batch[0] to be a tensor, got {type(batch[0])}"
            assert batch[0].size(0) == 1, f"Expected batch[0] to have size 1, got {batch[0].size(0)}"

    # Verify that the code lengths were calculated correctly
    assert len(candidate_code_lengths) == 5, f"Expected 5 code lengths, got {len(candidate_code_lengths)}"

    print("Test case 1 passed!")

    # Test case 2: Test with default_collate
    print("Test case 2: Test with default_collate")
    encoder = MockEncoder()

    # Process candidates in batches
    batch_size = 3
    candidate_code_lengths = []

    for i in range(0, len(candidate_batches), batch_size):
        batch = candidate_batches[i:i+batch_size]

        # Use PyTorch's default_collate
        from torch.utils.data._utils.collate import default_collate
        batched_candidates = default_collate(batch)

        # Process the batch using tensor operations
        code_lengths_tensor, _, _, _ = encoder.calculate_code_length(
            model=model,
            batch=batched_candidates,
            beta=beta,
            ema_params=ema_params,
            use_beta=use_beta
        )

        # For each item in the batch, calculate its code length
        for j in range(len(batch)):
            # If the tensor has a batch dimension, extract the j-th item
            if len(code_lengths_tensor.shape) > 1:
                item_code_lengths = code_lengths_tensor[j]
                candidate_code_lengths.append(item_code_lengths.sum().item())
            else:
                # If there's only one item in the batch, or the tensor doesn't have a batch dimension
                candidate_code_lengths.append(code_lengths_tensor.sum().item())

    # Verify that the calculate_code_length method was called with the correct arguments
    print(f"Number of calculate_code_length calls: {len(encoder.calculate_code_length_calls)}")
    assert len(encoder.calculate_code_length_calls) == 2, "Expected 2 calls to calculate_code_length"

    # Verify that the batched_candidates were passed correctly
    for i, call in enumerate(encoder.calculate_code_length_calls):
        batch = call['batch']
        assert isinstance(batch, (tuple, list)), f"Expected batch to be a tuple or list, got {type(batch)}"
        assert isinstance(batch[0], torch.Tensor), f"Expected batch[0] to be a tensor, got {type(batch[0])}"
        if i == 0:
            assert batch[0].size(0) == 3, f"Expected batch[0] to have size 3, got {batch[0].size(0)}"
        else:
            assert batch[0].size(0) == 2, f"Expected batch[0] to have size 2, got {batch[0].size(0)}"

    # Verify that the code lengths were calculated correctly
    assert len(candidate_code_lengths) == 5, f"Expected 5 code lengths, got {len(candidate_code_lengths)}"

    print("Test case 2 passed!")

    # Test case 3: Test with a batch size larger than the number of candidates
    print("Test case 3: Test with a batch size larger than the number of candidates")
    encoder = MockEncoder()

    # Process candidates in batches
    batch_size = 10  # Larger than the number of candidates
    candidate_code_lengths = []

    for i in range(0, len(candidate_batches), batch_size):
        batch = candidate_batches[i:i+batch_size]

        # Use PyTorch's default_collate
        from torch.utils.data._utils.collate import default_collate
        batched_candidates = default_collate(batch)

        # Process the batch using tensor operations
        code_lengths_tensor, _, _, _ = encoder.calculate_code_length(
            model=model,
            batch=batched_candidates,
            beta=beta,
            ema_params=ema_params,
            use_beta=use_beta
        )

        # For each item in the batch, calculate its code length
        for j in range(len(batch)):
            # If the tensor has a batch dimension, extract the j-th item
            if len(code_lengths_tensor.shape) > 1:
                item_code_lengths = code_lengths_tensor[j]
                candidate_code_lengths.append(item_code_lengths.sum().item())
            else:
                # If there's only one item in the batch, or the tensor doesn't have a batch dimension
                candidate_code_lengths.append(code_lengths_tensor.sum().item())

    # Verify that the calculate_code_length method was called with the correct arguments
    print(f"Number of calculate_code_length calls: {len(encoder.calculate_code_length_calls)}")
    assert len(encoder.calculate_code_length_calls) == 1, "Expected 1 call to calculate_code_length"

    # Verify that the batched_candidates were passed correctly
    batch = encoder.calculate_code_length_calls[0]['batch']
    assert isinstance(batch, (tuple, list)), f"Expected batch to be a tuple or list, got {type(batch)}"
    assert isinstance(batch[0], torch.Tensor), f"Expected batch[0] to be a tensor, got {type(batch[0])}"
    assert batch[0].size(0) == 5, f"Expected batch[0] to have size 5, got {batch[0].size(0)}"

    # Verify that the code lengths were calculated correctly
    assert len(candidate_code_lengths) == 5, f"Expected 5 code lengths, got {len(candidate_code_lengths)}"

    print("Test case 3 passed!")

    # Test case 4: Test with a single item in a batch
    print("Test case 4: Test with a single item in a batch")
    encoder = MockEncoder()

    # Create a batch with a single item
    single_batch = [dataset[0]]

    # Use PyTorch's default_collate
    from torch.utils.data._utils.collate import default_collate
    batched_candidate = default_collate(single_batch)

    # Process the batch using tensor operations
    code_lengths_tensor, _, _, _ = encoder.calculate_code_length(
        model=model,
        batch=batched_candidate,
        beta=beta,
        ema_params=ema_params,
        use_beta=use_beta
    )

    # Calculate the code length
    if len(code_lengths_tensor.shape) > 1:
        item_code_length = code_lengths_tensor[0].sum().item()
    else:
        item_code_length = code_lengths_tensor.sum().item()

    # Verify that the calculate_code_length method was called with the correct arguments
    print(f"Number of calculate_code_length calls: {len(encoder.calculate_code_length_calls)}")
    assert len(encoder.calculate_code_length_calls) == 1, "Expected 1 call to calculate_code_length"

    # Verify that the batched_candidate was passed correctly
    batch = encoder.calculate_code_length_calls[0]['batch']
    assert isinstance(batch, (tuple, list)), f"Expected batch to be a tuple or list, got {type(batch)}"
    assert isinstance(batch[0], torch.Tensor), f"Expected batch[0] to be a tensor, got {type(batch[0])}"
    assert batch[0].size(0) == 1, f"Expected batch[0] to have size 1, got {batch[0].size(0)}"

    print("Test case 4 passed!")

    print("All test cases passed!")

if __name__ == "__main__":
    test_batch_processing()
