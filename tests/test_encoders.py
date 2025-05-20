import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader

# Import directly from the package
from preqtorch import BlockEncoder, MIRSEncoder

# Define a simple character-level model for the Spanish phonetic transcription task
class SimplePhoneticModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimplePhoneticModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_size = output_size
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x, target_in=None):
        # Handle device placement in the forward method
        device = next(self.parameters()).device

        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            if isinstance(x, int):
                x = torch.tensor([x], dtype=torch.long, device=device)
            elif isinstance(x, list):
                x = torch.tensor(x, dtype=torch.long, device=device)

        # Move tensor to the correct device if needed
        if hasattr(x, 'to'):
            x = x.to(device)

        # Ensure x has at least 2 dimensions [batch_size, seq_len]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Ensure indices are within bounds
        input_size = self.embedding.num_embeddings
        x = torch.clamp(x, 0, input_size - 1)

        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)
        # embedded shape: [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(embedded)
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        output = self.fc(lstm_out)
        # output shape: [batch_size, seq_len, output_size]

        # Ensure output has 3 dimensions [batch_size, seq_len, output_size]
        if output.dim() == 2:
            output = output.unsqueeze(1)  # Add sequence dimension if missing

        return output

# Define a dataset for the Spanish phonetic transcription task
class SpanishPhoneticDataset(Dataset):
    def __init__(self, file_path, max_samples=1000):
        self.data = []
        self.char_to_idx = {'<pad>': 0, '<bos>': 1}
        self.phoneme_to_idx = {'<pad>': 0, '<bos>': 1}

        # Read the data file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Limit the number of samples for faster testing
        lines = lines[:max_samples]

        # Process each line
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word, phonemes = parts

                # Create character indices for the word
                for char in word:
                    if char not in self.char_to_idx:
                        self.char_to_idx[char] = len(self.char_to_idx)

                # Create phoneme indices
                phoneme_list = phonemes.split()
                for phoneme in phoneme_list:
                    if phoneme not in self.phoneme_to_idx:
                        self.phoneme_to_idx[phoneme] = len(self.phoneme_to_idx)

                self.data.append((word, phoneme_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word, phonemes = self.data[idx]

        # Convert word to tensor of indices
        word_indices = [self.char_to_idx.get(char, 0) for char in word]
        word_tensor = torch.tensor(word_indices, dtype=torch.long)

        # Convert phonemes to tensor of indices
        phoneme_indices = [self.phoneme_to_idx.get(phoneme, 0) for phoneme in phonemes]
        phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long)

        # Create a mask for the target (all True in this case)
        mask = torch.ones_like(phoneme_tensor, dtype=torch.bool)

        return word_tensor, phoneme_tensor, mask

# Collate function to handle variable length sequences
def collate_fn(batch):
    # Sort the batch by word length (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    # Get the data
    words, phonemes, masks = zip(*batch)

    # Pad the sequences
    words_padded = nn.utils.rnn.pad_sequence(words, batch_first=True)
    phonemes_padded = nn.utils.rnn.pad_sequence(phonemes, batch_first=True)
    masks_padded = nn.utils.rnn.pad_sequence(masks, batch_first=True)

    return words_padded, phonemes_padded, masks_padded

# Custom loss function
def phonetic_loss_fn(outputs, targets, mask):
    # Remove debug print statements for clarity

    # Ensure outputs is a tensor
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(f"Expected outputs to be a tensor, got {type(outputs)}")

    # Ensure targets is a tensor
    if not isinstance(targets, torch.Tensor):
        if isinstance(targets, tuple) and len(targets) > 0:
            targets = targets[0]  # Take the first element if it's a tuple
            if not isinstance(targets, torch.Tensor):
                raise TypeError(f"Expected targets[0] to be a tensor, got {type(targets)}")
        else:
            raise TypeError(f"Expected targets to be a tensor, got {type(targets)}")

    # Ensure mask is a tensor or convert it to one
    if not isinstance(mask, torch.Tensor):
        if isinstance(mask, tuple) and len(mask) > 0:
            mask = mask[0]  # Take the first element if it's a tuple
            if not isinstance(mask, torch.Tensor):
                if isinstance(mask, bool):
                    # Create a tensor with all True values with the same shape as targets
                    mask = torch.ones_like(targets, dtype=torch.bool)
                else:
                    raise TypeError(f"Expected mask[0] to be a tensor or bool, got {type(mask)}")
        elif isinstance(mask, bool):
            # Create a tensor with all True values with the same shape as targets
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            raise TypeError(f"Expected mask to be a tensor or bool, got {type(mask)}")

    # Get the dimensions
    batch_size, seq_len, output_size = outputs.shape

    # If the batch sizes don't match, we need to handle this case
    if batch_size != targets.size(0) or batch_size != mask.size(0):
        # If outputs has batch size 1, we need to repeat it to match the batch size of targets and mask
        if batch_size == 1:
            outputs = outputs.repeat(targets.size(0), 1, 1)
            batch_size = outputs.shape[0]
        # If targets has batch size 1, we need to repeat it to match the batch size of outputs
        elif targets.size(0) == 1:
            targets = targets.repeat(batch_size, 1)
        # If mask has batch size 1, we need to repeat it to match the batch size of outputs
        elif mask.size(0) == 1:
            mask = mask.repeat(batch_size, 1)
        else:
            # If the batch sizes are different and neither is 1, we can't handle this case
            raise ValueError(f"Batch size mismatch: outputs {batch_size}, targets {targets.size(0)}, mask {mask.size(0)}")

    # Make sure targets and mask have the right shape
    if targets.dim() == 2:
        targets_seq_len = targets.size(1)
    else:
        targets_seq_len = 1

    if mask.dim() == 2:
        mask_seq_len = mask.size(1)
    else:
        mask_seq_len = 1

    # Ensure all sequences have the same length by padding or truncating
    min_seq_len = min(seq_len, targets_seq_len, mask_seq_len)

    # Truncate if necessary
    outputs = outputs[:, :min_seq_len, :]
    if targets.dim() == 2:
        targets = targets[:, :min_seq_len]
    if mask.dim() == 2:
        mask = mask[:, :min_seq_len]

    # Reshape outputs to [batch_size * min_seq_len, output_size]
    outputs = outputs.reshape(-1, output_size)

    # Reshape targets to [batch_size * min_seq_len]
    targets = targets.reshape(-1)

    # Reshape mask to [batch_size * min_seq_len]
    mask = mask.reshape(-1)

    # Apply the mask
    masked_outputs = outputs[mask]
    masked_targets = targets[mask]

    # Check if target indices are within bounds
    output_size = masked_outputs.size(-1)
    if torch.any(masked_targets >= output_size):
        raise ValueError(f"Error: Target indices out of bounds. Target max: {masked_targets.max().item()}, Output size: {output_size}. The model's output size is too small for the dataset.")

    # Calculate cross entropy loss
    return F.cross_entropy(masked_outputs, masked_targets, reduction='sum')

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Path to the Spanish phonetic data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")

    # Create the dataset
    dataset = SpanishPhoneticDataset(data_path, max_samples=500)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(dataset.phoneme_to_idx)}")

    # Test BlockEncoder
    print("\nTesting BlockEncoder...")
    block_encoder = BlockEncoder(
        model_class=lambda: SimplePhoneticModel(
            input_size=len(dataset.char_to_idx),
            hidden_size=64,
            output_size=len(dataset.phoneme_to_idx)
        ),
        loss_fn=phonetic_loss_fn
    )

    # Encode with BlockEncoder (default parameters)
    model, code_length = block_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (Block)",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        stop_points=[0.5, 1.0],
        patience=5,
        collate_fn=collate_fn,
        use_device_handling=False
    )

    print(f"Block Encoder - Code length: {code_length}.")

    # Test BlockEncoder with shuffle=False
    print("\nTesting BlockEncoder with shuffle=False...")
    model, code_length = block_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (Block, no shuffle)",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        stop_points=[0.5, 1.0],
        patience=5,
        collate_fn=collate_fn,
        use_device_handling=False,
        shuffle=False
    )

    print(f"Block Encoder (no shuffle) - Code length: {code_length}.")

    # Test BlockEncoder with return_code_length_history=True
    print("\nTesting BlockEncoder with return_code_length_history=True...")
    model, code_length, code_length_history = block_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (Block, with history)",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        stop_points=[0.5, 1.0],
        patience=5,
        collate_fn=collate_fn,
        use_device_handling=False,
        return_code_length_history=True
    )

    print(f"Block Encoder (with history) - Code length: {code_length}.")
    print(f"Block Encoder - Code length history: {code_length_history}")

    # Test BlockEncoder with num_samples=None
    print("\nTesting BlockEncoder with num_samples=None...")
    model, code_length = block_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (Block, full dataset)",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        stop_points=[0.5, 1.0],
        patience=5,
        collate_fn=collate_fn,
        use_device_handling=False,
        num_samples=None
    )

    print(f"Block Encoder (full dataset) - Code length: {code_length}.")

    # Test MIRSEncoder
    print("\nTesting MIRSEncoder...")
    mirs_encoder = MIRSEncoder(
        model_class=lambda: SimplePhoneticModel(
            input_size=len(dataset.char_to_idx),
            hidden_size=64,
            output_size=len(dataset.phoneme_to_idx)
        ),
        loss_fn=phonetic_loss_fn
    )

    # Encode with MIRSEncoder
    model, code_length, ema_params, beta, replay_streams = mirs_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (MIRS)",
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn,
        use_device_handling=False,
        use_beta=True,
        use_ema=True
    )

    print(f"MIRS Encoder - Code length: {code_length}.")

    # Test MIRSEncoder without beta and EMA
    print("\nTesting MIRSEncoder without beta and EMA...")
    mirs_encoder_no_beta_ema = MIRSEncoder(
        model_class=lambda: SimplePhoneticModel(
            input_size=len(dataset.char_to_idx),
            hidden_size=64,
            output_size=len(dataset.phoneme_to_idx)
        ),
        loss_fn=phonetic_loss_fn
    )

    # Encode with MIRSEncoder without beta and EMA
    model, code_length, ema_params, beta, replay_streams = mirs_encoder_no_beta_ema.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (MIRS no beta/EMA)",
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn,
        use_device_handling=False,
        use_beta=False,
        use_ema=False
    )

    print(f"MIRS Encoder (no beta/EMA) - Code length: {code_length}.")

    # Test MIRSEncoder with shuffle=False
    print("\nTesting MIRSEncoder with shuffle=False...")
    mirs_encoder_no_shuffle = MIRSEncoder(
        model_class=lambda: SimplePhoneticModel(
            input_size=len(dataset.char_to_idx),
            hidden_size=64,
            output_size=len(dataset.phoneme_to_idx)
        ),
        loss_fn=phonetic_loss_fn
    )

    # Encode with MIRSEncoder with shuffle=False
    model, code_length, ema_params, beta, replay_streams = mirs_encoder_no_shuffle.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (MIRS, no shuffle)",
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn,
        use_device_handling=False,
        use_beta=True,
        use_ema=True,
        shuffle=False
    )

    print(f"MIRS Encoder (no shuffle) - Code length: {code_length}.")

    # Test MIRSEncoder with return_code_length_history=True
    print("\nTesting MIRSEncoder with return_code_length_history=True...")
    mirs_encoder_with_history = MIRSEncoder(
        model_class=lambda: SimplePhoneticModel(
            input_size=len(dataset.char_to_idx),
            hidden_size=64,
            output_size=len(dataset.phoneme_to_idx)
        ),
        loss_fn=phonetic_loss_fn
    )

    # Encode with MIRSEncoder with return_code_length_history=True
    model, code_length, code_length_history, ema_params, beta, replay_streams = mirs_encoder_with_history.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (MIRS, with history)",
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn,
        use_device_handling=False,
        use_beta=True,
        use_ema=True,
        return_code_length_history=True
    )

    print(f"MIRS Encoder (with history) - Code length: {code_length}.")
    print(f"MIRS Encoder - Code length history: {code_length_history}")

    # Test MIRSEncoder with num_samples=None
    print("\nTesting MIRSEncoder with num_samples=None...")
    mirs_encoder_full_dataset = MIRSEncoder(
        model_class=lambda: SimplePhoneticModel(
            input_size=len(dataset.char_to_idx),
            hidden_size=64,
            output_size=len(dataset.phoneme_to_idx)
        ),
        loss_fn=phonetic_loss_fn
    )

    # Encode with MIRSEncoder with num_samples=None
    model, code_length, ema_params, beta, replay_streams = mirs_encoder_full_dataset.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (MIRS, full dataset)",
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn,
        use_device_handling=False,
        use_beta=True,
        use_ema=True,
        num_samples=None
    )

    print(f"MIRS Encoder (full dataset) - Code length: {code_length}.")

if __name__ == "__main__":
    main()
