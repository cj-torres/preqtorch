import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

# Import directly from the package
from preqtorch import MIRSEncoder, prequential_clustering

# Import the SimplePhoneticModel, SpanishPhoneticDataset, and phonetic_loss_fn from test_encoders.py
from test_encoders import SimplePhoneticModel, SpanishPhoneticDataset, collate_fn, phonetic_loss_fn

# # Copy of the prequential_clustering function from preqcluster.py with absolute import
# def prequential_clustering(encoder, dataset, beam_width=5, learning_rate=0.001, seed=42, alpha=0.1,
#                           batch_size=32, n_replay_streams=2, use_beta=True, use_ema=True,
#                           collate_fn=None, use_device_handling=False):
#     """
#     Performs prequential clustering using a MIRS prequential encoder.
#
#     Args:
#         encoder: MIRSEncoder instance.
#         dataset: Dataset to cluster.
#         beam_width: Width of the beam for greedy beam search.
#         learning_rate: Learning rate for optimization.
#         seed: Random seed for reproducibility.
#         alpha: EMA smoothing factor.
#         batch_size: Batch size for training.
#         n_replay_streams: Number of replay streams for MIRS encoder.
#         use_beta: Whether to use beta scaling.
#         use_ema: Whether to use EMA.
#         collate_fn: Function to collate data into batches.
#         use_device_handling: Whether to use device handling.
#
#     Returns:
#         best_sequence: List[int] - best order of indices.
#         best_code_lengths: List[float] - corresponding code lengths.
#         clusters: List[List[int]] - clusters of data points.
#     """
#     # Enable anomaly detection to help diagnose runtime errors
#     torch.autograd.set_detect_anomaly(True)
#
#     torch.manual_seed(seed)
#     random.seed(seed)
#
#     n = len(dataset)
#
#     # Randomly sample a nucleation datapoint to start with
#     start_idx = random.randint(0, n-1)
#
#     # Initialize a single model with the nucleation datapoint
#     single_item_dataset = [dataset[start_idx]]
#
#     # Encode the single item
#     model, code_len, ema_params, beta, replay_streams = encoder.encode(
#         dataset=single_item_dataset,
#         set_name=f"Initial Item {start_idx}",
#         n_replay_streams=n_replay_streams,
#         learning_rate=learning_rate,
#         batch_size=1,  # Single item
#         seed=seed,
#         alpha=alpha,
#         collate_fn=collate_fn,
#         use_device_handling=use_device_handling,
#         use_beta=use_beta,
#         use_ema=use_ema,
#         shuffle=False  # No need to shuffle a single item
#     )
#
#     # Initialize the first beam
#     beams = [
#         (([start_idx], [code_len], {start_idx}), (model, ema_params, beta, replay_streams))
#     ]
#
#     for step in tqdm(range(1, n), desc='Greedy beam search'):
#         new_beams = []
#
#         # For each beam, find the best candidates to append
#         for (seq, code_lengths, used), (model, ema_params, beta, replay_streams) in beams:
#             # Collect all unused dataset items
#             unused_indices = [j for j in range(n) if j not in used]
#
#             if not unused_indices:
#                 # If all items are used in this beam, just keep it as is
#                 new_beams.append(((seq, code_lengths, used), (model, ema_params, beta, replay_streams), sum(code_lengths)))
#                 continue
#
#             # Create a batch of unused items for efficient prediction
#             candidate_batches = [dataset[j] for j in unused_indices]
#
#             # Calculate code lengths for all candidates using the current model with grad turned off
#             candidate_code_lengths = []
#
#             # Process candidates in large batches for efficiency
#             batch_size_local = 32  # Choose an appropriate batch size
#             for i in range(0, len(candidate_batches), batch_size_local):
#                 batch = candidate_batches[i:i+batch_size_local]
#
#                 # Use the provided collate_fn or default collate function
#                 if collate_fn:
#                     # Apply the provided collate function to create a properly formatted batch
#                     batched_candidates = collate_fn(batch)
#                 else:
#                     # Use PyTorch's default_collate if no collate_fn is provided
#                     from torch.utils.data._utils.collate import default_collate
#                     batched_candidates = default_collate(batch)
#
#                 with torch.no_grad():
#                     # Process the batch using tensor operations
#                     code_lengths_tensor, _, _, _ = encoder.calculate_code_length(
#                         model=model,
#                         batch=batched_candidates,
#                         beta=beta,
#                         ema_params=ema_params,
#                         use_beta=use_beta
#                     )
#
#                     # For each item in the batch, calculate its code length
#                     for j in range(len(batch)):
#                         # If the tensor has a batch dimension, extract the j-th item
#                         if len(code_lengths_tensor.shape) > 1:
#                             item_code_lengths = code_lengths_tensor[j]
#                             candidate_code_lengths.append(item_code_lengths.sum().item())
#                         else:
#                             # If there's only one item in the batch, or the tensor doesn't have a batch dimension
#                             candidate_code_lengths.append(code_lengths_tensor.sum().item())
#
#             # Create (index, code_length) pairs and sort by code length
#             candidates = list(zip(unused_indices, candidate_code_lengths))
#             candidates.sort(key=lambda x: x[1])  # Sort by code length
#
#             # Take the top beam_width candidates
#             top_candidates = candidates[:beam_width]
#
#             # Train a model for each top candidate
#             for j, candidate_code_len in top_candidates:
#                 # Create a dataset with just the new item
#                 single_item_dataset = [dataset[j]]
#
#                 # Clone model and state for this branch
#                 new_model = deepcopy(model)
#                 new_ema_params = {k: v.clone().detach() for k, v in ema_params.items()}
#                 new_beta = torch.nn.Parameter(beta.clone().detach())
#                 # We'll use the same replay_streams object
#
#                 # Encode with existing model
#                 new_model, new_code_len, new_ema_params, new_beta, new_replay_streams = encoder.encode_with_model(
#                     model=new_model,
#                     ema_params=new_ema_params,
#                     beta=new_beta,
#                     dataset=single_item_dataset,
#                     set_name=f"Item {j}",
#                     n_replay_streams=n_replay_streams,
#                     learning_rate=learning_rate,
#                     batch_size=1,  # Single item
#                     seed=seed,
#                     alpha=alpha,
#                     collate_fn=collate_fn,
#                     use_device_handling=use_device_handling,
#                     use_beta=use_beta,
#                     use_ema=use_ema,
#                     shuffle=False,  # No need to shuffle a single item
#                     replay_streams=replay_streams  # Pass the existing replay_streams
#                 )
#
#                 new_seq = seq + [j]
#                 new_code_lengths = code_lengths + [new_code_len]
#                 new_used = used | {j}
#
#                 total_code = sum(new_code_lengths)
#                 new_beams.append(((new_seq, new_code_lengths, new_used),
#                                   (new_model, new_ema_params, new_beta, new_replay_streams),
#                                   total_code))
#
#         # If we have no new beams (all items used), keep the old ones
#         if not new_beams:
#             break
#
#         new_beams.sort(key=lambda x: x[2])  # sort by total code length
#         beams = [(x[0], x[1]) for x in new_beams[:beam_width]]
#
#     best_seq, best_code_lengths, _ = min(
#         [(seq, code_lengths, sum(code_lengths)) for (seq, code_lengths, _) in [b[0] for b in beams]],
#         key=lambda x: x[2]
#     )
#
#     # Detect clusters using code length boundaries
#     boundaries = detect_codelength_boundaries(best_code_lengths)
#     clusters = []
#     start_idx = 0
#
#     for boundary in boundaries:
#         clusters.append(best_seq[start_idx:boundary])
#         start_idx = boundary
#
#     # Add the last cluster
#     if start_idx < len(best_seq):
#         clusters.append(best_seq[start_idx:])
#
#     return best_seq, best_code_lengths, clusters
#
# # Copy of the detect_codelength_boundaries function from preqcluster.py
# def detect_codelength_boundaries(code_lengths, window_size=5, std_threshold=2.0):
#     """
#     Detects boundaries in code lengths to identify clusters.
#
#     Args:
#         code_lengths: List of code lengths.
#         window_size: Size of the smoothing window.
#         std_threshold: Threshold for boundary detection in standard deviations.
#
#     Returns:
#         boundaries: List of indices where boundaries are detected.
#     """
#     code_lengths = np.array(code_lengths)
#     smoothed = np.convolve(code_lengths, np.ones(window_size)/window_size, mode='valid')
#     delta = np.diff(smoothed)
#     delta = np.concatenate(([0] * window_size, delta))
#
#     mean_delta = np.mean(delta)
#     std_delta = np.std(delta)
#     threshold = mean_delta + std_threshold * std_delta
#
#     boundaries = [i for i in range(1, len(delta)) if delta[i] > threshold]
#     return boundaries

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Path to the Spanish phonetic data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")

    # Create the dataset with a larger portion of data (2000 samples)
    dataset = SpanishPhoneticDataset(data_path, max_samples=2000)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(dataset.phoneme_to_idx)}")

    # Create a MIRS encoder with a SimplePhoneticModel
    mirs_encoder = MIRSEncoder(
        model_class=lambda: SimplePhoneticModel(
            input_size=len(dataset.char_to_idx),
            hidden_size=64,
            output_size=len(dataset.phoneme_to_idx)
        ),
        loss_fn=phonetic_loss_fn
    )

    print("\nPerforming prequential clustering...")


    # Use a smaller beam width and dataset size for faster execution
    beam_width = 3

    # Perform prequential clustering
    best_seq, best_code_lengths, clusters = prequential_clustering(
        encoder=mirs_encoder,
        dataset=dataset,
        beam_width=beam_width,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        batch_size=32,
        n_replay_streams=2,
        use_beta=True,
        use_ema=True,
        collate_fn=collate_fn,
        use_device_handling=False
    )

    print(f"\nClustering complete. Found {len(clusters)} clusters.")

    # Print the first 10 words of each cluster
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1} (size: {len(cluster)}):")
        print("First 10 words:")
        for j in range(min(10, len(cluster))):
            idx = cluster[j]
            word, _ = dataset.data[idx]
            print(f"  {j+1}. {word}")

if __name__ == "__main__":
    main()
