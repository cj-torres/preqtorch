import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .encoders import MIRSEncoder

# ------------------------------
# Prequential Clustering
# ------------------------------
def prequential_clustering(encoder, dataset, beam_width=5, learning_rate=0.001, seed=42, alpha=0.1, 
                          batch_size=32, n_replay_streams=2, use_beta=True, use_ema=True, 
                          collate_fn=None, use_device_handling=False):
    """
    Performs prequential clustering using a MIRS prequential encoder.

    This implementation:
    1. Initializes a single model and randomly samples a nucleation datapoint
    2. Predicts code lengths for new dataset items using batch processing with grad turned off
    3. Identifies N candidate dataset items to append (N = beam_width)
    4. Trains N models on these continuations
    5. Repeats, maintaining the beams until the dataset is exhausted

    Args:
        encoder: MIRSEncoder instance.
        dataset: Dataset to cluster.
        beam_width: Width of the beam for greedy beam search.
        learning_rate: Learning rate for optimization.
        seed: Random seed for reproducibility.
        alpha: EMA smoothing factor.
        batch_size: Batch size for training.
        n_replay_streams: Number of replay streams for MIRS encoder.
        use_beta: Whether to use beta scaling.
        use_ema: Whether to use EMA.
        collate_fn: Function to collate data into batches.
        use_device_handling: Whether to use device handling.

    Returns:
        best_sequence: List[int] - best order of indices.
        best_code_lengths: List[float] - corresponding code lengths.
        clusters: List[List[int]] - clusters of data points.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    n = len(dataset)

    # Randomly sample a nucleation datapoint to start with
    start_idx = random.randint(0, n-1)

    # Initialize a single model with the nucleation datapoint
    single_item_dataset = [dataset[start_idx]]

    # Encode the single item
    model, code_len, ema_params, beta, replay_streams = encoder.encode(
        dataset=single_item_dataset,
        set_name=f"Initial Item {start_idx}",
        n_replay_streams=n_replay_streams,
        learning_rate=learning_rate,
        batch_size=1,  # Single item
        seed=seed,
        alpha=alpha,
        collate_fn=collate_fn,
        use_device_handling=use_device_handling,
        use_beta=use_beta,
        use_ema=use_ema,
        shuffle=False  # No need to shuffle a single item
    )

    # Initialize the first beam
    beams = [
        (([start_idx], [code_len], {start_idx}), (model, ema_params, beta, replay_streams))
    ]

    for step in tqdm(range(1, n), desc='Greedy beam search'):
        new_beams = []

        # For each beam, find the best candidates to append
        for (seq, code_lengths, used), (model, ema_params, beta, replay_streams) in beams:
            # Collect all unused dataset items
            unused_indices = [j for j in range(n) if j not in used]

            if not unused_indices:
                # If all items are used in this beam, just keep it as is
                new_beams.append(((seq, code_lengths, used), (model, ema_params, beta, replay_streams), sum(code_lengths)))
                continue

            # Create a batch of unused items for efficient prediction
            candidate_batches = [dataset[j] for j in unused_indices]

            # Calculate code lengths for all candidates using the current model with grad turned off
            candidate_code_lengths = []

            # Process candidates in large batches for efficiency
            batch_size_local = 32  # Choose an appropriate batch size
            for i in range(0, len(candidate_batches), batch_size_local):
                batch = candidate_batches[i:i+batch_size_local]

                # Use the provided collate_fn or default collate function
                if collate_fn:
                    # Apply the provided collate function to create a properly formatted batch
                    batched_candidates = collate_fn(batch)
                else:
                    # Use PyTorch's default_collate if no collate_fn is provided
                    from torch.utils.data._utils.collate import default_collate
                    batched_candidates = default_collate(batch)

                with torch.no_grad():
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

            # Create (index, code_length) pairs and sort by code length
            candidates = list(zip(unused_indices, candidate_code_lengths))
            candidates.sort(key=lambda x: x[1])  # Sort by code length

            # Take the top beam_width candidates
            top_candidates = candidates[:beam_width]

            # Train a model for each top candidate
            for j, candidate_code_len in top_candidates:
                # Create a dataset with just the new item
                single_item_dataset = [dataset[j]]

                # Clone model and state for this branch
                new_model = deepcopy(model)
                new_ema_params = {k: v.clone().detach() for k, v in ema_params.items()}
                new_beta = torch.nn.Parameter(beta.clone().detach())
                # We'll use the same replay_streams object

                # Encode with existing model
                new_model, new_code_len, new_ema_params, new_beta, new_replay_streams = encoder.encode_with_model(
                    model=new_model,
                    ema_params=new_ema_params,
                    beta=new_beta,
                    dataset=single_item_dataset,
                    set_name=f"Item {j}",
                    n_replay_streams=n_replay_streams,
                    learning_rate=learning_rate,
                    batch_size=1,  # Single item
                    seed=seed,
                    alpha=alpha,
                    collate_fn=collate_fn,
                    use_device_handling=use_device_handling,
                    use_beta=use_beta,
                    use_ema=use_ema,
                    shuffle=False,  # No need to shuffle a single item
                    replay_streams=replay_streams  # Pass the existing replay_streams
                )

                new_seq = seq + [j]
                new_code_lengths = code_lengths + [new_code_len]
                new_used = used | {j}

                total_code = sum(new_code_lengths)
                new_beams.append(((new_seq, new_code_lengths, new_used), 
                                  (new_model, new_ema_params, new_beta, new_replay_streams), 
                                  total_code))

        # If we have no new beams (all items used), keep the old ones
        if not new_beams:
            break

        new_beams.sort(key=lambda x: x[2])  # sort by total code length
        beams = [(x[0], x[1]) for x in new_beams[:beam_width]]

    best_seq, best_code_lengths, _ = min(
        [(seq, code_lengths, sum(code_lengths)) for (seq, code_lengths, _) in [b[0] for b in beams]],
        key=lambda x: x[2]
    )

    # Detect clusters using code length boundaries
    boundaries = detect_codelength_boundaries(best_code_lengths)
    clusters = []
    start_idx = 0

    for boundary in boundaries:
        clusters.append(best_seq[start_idx:boundary])
        start_idx = boundary

    # Add the last cluster
    if start_idx < len(best_seq):
        clusters.append(best_seq[start_idx:])

    return best_seq, best_code_lengths, clusters

# ------------------------------
# Change Point Detection
# ------------------------------
def detect_codelength_boundaries(code_lengths, window_size=5, std_threshold=2.0):
    """
    Detects boundaries in code lengths to identify clusters.

    Args:
        code_lengths: List of code lengths.
        window_size: Size of the smoothing window.
        std_threshold: Threshold for boundary detection in standard deviations.

    Returns:
        boundaries: List of indices where boundaries are detected.
    """
    code_lengths = np.array(code_lengths)
    smoothed = np.convolve(code_lengths, np.ones(window_size)/window_size, mode='valid')
    delta = np.diff(smoothed)
    delta = np.concatenate(([0] * window_size, delta))

    mean_delta = np.mean(delta)
    std_delta = np.std(delta)
    threshold = mean_delta + std_threshold * std_delta

    boundaries = [i for i in range(1, len(delta)) if delta[i] > threshold]
    return boundaries


def consensus_prequential_clustering(encoder, dataset, num_runs=10, boundary_fn=detect_codelength_boundaries):
    """
    Repeatedly runs prequential encoding with shuffled data to build a consensus co-clustering matrix.

    Args:
        encoder: A prequential encoder (e.g. MIRSEncoder) with encode(..., return_code_length_history=True)
        dataset: A list of individual items (batches)
        num_runs: Number of random restart runs
        boundary_fn: Callable to identify change points given a sequence of code lengths

    Returns:
        co_matrix: (n x n) numpy array where entry (i,j) is the proportion of runs
                   where items i and j were in the same cluster.
    """
    n = len(dataset)
    co_matrix = np.zeros((n, n))

    for run in range(num_runs):
        perm = random.sample(range(n), n)
        shuffled = [dataset[i] for i in perm]

        model, total_cl, code_lengths, *_ = encoder.encode(
            shuffled,
            set_name=f'run_{run}',
            return_code_length_history=True
        )

        boundaries = boundary_fn(code_lengths)

        clusters = []
        prev = 0
        for b in boundaries:
            clusters.append(perm[prev:b])
            prev = b
        clusters.append(perm[prev:])

        for cluster in clusters:
            for i in cluster:
                for j in cluster:
                    co_matrix[i, j] += 1

    co_matrix /= num_runs
    return co_matrix


def crp_clustering(encoder_class, dataset, alpha=1.0, max_iter=10, seed=42, use_crp=True):
    """
    Cluster data using the Minimum Description Length principle.

    Args:
        encoder_class: an instance of a prequential encoder
        dataset: list of preprocessed batches
        alpha: CRP concentration parameter (only used if use_crp is True)
        max_iter: number of EM iterations
        seed: random seed
        use_crp: if True, allow new cluster creation via Chinese Restaurant Process

    Returns:
        cluster_assignments: list of cluster ids
    """
    random.seed(seed)
    torch.manual_seed(seed)

    n = len(dataset)
    clusters = {0: [i for i in range(n)]}  # initially one cluster
    assignments = [0] * n

    for iteration in range(max_iter):
        new_assignments = []
        models = {}
        model_stats = {}

        # Train a model for each cluster
        for cid, indices in clusters.items():
            sub_data = [dataset[i] for i in indices]
            encoder = deepcopy(encoder_class)
            model, codelength, *_ = encoder.encode(
                sub_data, set_name=f'cluster_{cid}', return_code_length_history=False
            )
            models[cid] = (encoder, model)
            model_stats[cid] = (codelength, len(sub_data))

        for i, item in enumerate(dataset):
            logps = []
            for cid, (encoder, model) in models.items():
                encoder_copy = deepcopy(encoder)
                enc, item_cl, *_ = encoder_copy.encode_with_model(
                    model, *encoder.encode(
                        [item], set_name=f'test_{i}_{cid}',
                        return_code_length_history=False
                    )[3:]
                )
                logps.append((item_cl, cid))

            # Optionally consider a new cluster via CRP
            if use_crp:
                new_cid = max(clusters) + 1
                encoder = deepcopy(encoder_class)
                model, cl_new, *_ = encoder.encode(
                    [item], set_name=f'new_cluster_{i}', return_code_length_history=False
                )
                crp_logp = cl_new - np.log(alpha / (alpha + n - 1))
                logps.append((crp_logp, new_cid))

            best_cl, best_cid = min(logps)
            new_assignments.append(best_cid)

        # Update clusters
        new_clusters = {}
        for i, cid in enumerate(new_assignments):
            if cid not in new_clusters:
                new_clusters[cid] = []
            new_clusters[cid].append(i)

        clusters = new_clusters
        assignments = new_assignments

    return assignments