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
from preqtorch import MIRSEncoder, crp_clustering

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

    print("\nPerforming CRP clustering...")

    # Use fewer iterations for faster execution
    max_iter = 100

    # Perform CRP clustering
    assignments = crp_clustering(
        encoder_class=mirs_encoder,
        dataset=dataset,
        alpha=1.0,
        max_iter=max_iter,
        seed=42,
        use_crp=True,
        n_replay_streams=5,
        learning_rate=0.001,
        batch_size=32,
        collate_fn=collate_fn,
        use_device_handling=True,
        use_beta=True,
        use_ema=True
    )

    print(f"\nClustering complete. Found {len(set(assignments))} clusters.")

    # Convert assignments to clusters
    clusters = {}
    for i, cid in enumerate(assignments):
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(i)

    # Print the clusters
    print(f"\nCluster details:")
    for cid, indices in clusters.items():
        print(f"\nCluster {cid+1} (size: {len(indices)}):")
        print("First 5 words:")
        for j in range(min(5, len(indices))):
            idx = indices[j]
            word, _ = dataset.data[idx]
            print(f"  {j+1}. {word}")

    # Test without CRP
    print("\nPerforming clustering without CRP...")

    assignments_no_crp = crp_clustering(
        encoder_class=mirs_encoder,
        dataset=dataset,
        alpha=1.0,
        max_iter=max_iter,
        seed=42,
        use_crp=False,
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        collate_fn=collate_fn,
        use_device_handling=True,
        use_beta=True,
        use_ema=True
    )

    print(f"\nClustering without CRP complete. Found {len(set(assignments_no_crp))} clusters.")

    # Convert assignments to clusters
    clusters_no_crp = {}
    for i, cid in enumerate(assignments_no_crp):
        if cid not in clusters_no_crp:
            clusters_no_crp[cid] = []
        clusters_no_crp[cid].append(i)

    # Print the clusters
    print(f"\nCluster details (without CRP):")
    for cid, indices in clusters_no_crp.items():
        print(f"\nCluster {cid+1} (size: {len(indices)}):")
        print("First 5 words:")
        for j in range(min(5, len(indices))):
            idx = indices[j]
            word, _ = dataset.data[idx]
            print(f"  {j+1}. {word}")

if __name__ == "__main__":
    main()
