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
from preqtorch import MIRSEncoder, consensus_prequential_clustering, detect_codelength_boundaries

# Import the SimplePhoneticModel, SpanishPhoneticDataset, and phonetic_loss_fn from test_encoders.py
from test_encoders import SimplePhoneticModel, SpanishPhoneticDataset, collate_fn, phonetic_loss_fn

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Path to the Spanish phonetic data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")

    # Create the dataset with a smaller portion of data for faster testing
    dataset = SpanishPhoneticDataset(data_path, max_samples=100)

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

    print("\nPerforming consensus prequential clustering...")

    # Use a smaller number of runs for faster execution
    num_runs = 3

    # Perform consensus prequential clustering
    co_matrix = consensus_prequential_clustering(
        encoder=mirs_encoder,
        dataset=dataset,
        num_runs=num_runs,
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn,
        use_device_handling=True,
        use_beta=True,
        use_ema=True
    )

    print(f"\nClustering complete. Co-clustering matrix shape: {co_matrix.shape}")

    # Analyze the co-clustering matrix
    print("\nAnalyzing co-clustering matrix:")
    print(f"Mean co-clustering value: {np.mean(co_matrix)}")
    print(f"Max co-clustering value: {np.max(co_matrix)}")
    print(f"Min co-clustering value: {np.min(co_matrix)}")

    # Find clusters using a simple threshold
    threshold = 0.7  # Items that appear together in at least 70% of runs
    clusters = []
    visited = set()

    for i in range(len(dataset)):
        if i in visited:
            continue

        cluster = [i]
        visited.add(i)

        for j in range(len(dataset)):
            if j != i and j not in visited and co_matrix[i, j] >= threshold:
                cluster.append(j)
                visited.add(j)

        if len(cluster) > 1:  # Only consider clusters with more than one item
            clusters.append(cluster)

    # Print the clusters
    print(f"\nFound {len(clusters)} clusters with threshold {threshold}:")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1} (size: {len(cluster)}):")
        print("First 5 words:")
        for j in range(min(5, len(cluster))):
            idx = cluster[j]
            word, _ = dataset.data[idx]
            print(f"  {j+1}. {word}")

if __name__ == "__main__":
    main()
