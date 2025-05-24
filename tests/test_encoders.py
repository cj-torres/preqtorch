import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader

# Import directly from the package
from preqtorch import BlockEncoder, MIREncoder, ModelClass

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

# Define a base dataset for the Spanish phonetic transcription task
class BaseSpanishPhoneticDataset(Dataset):
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

    def _get_tensors(self, idx):
        word, phonemes = self.data[idx]

        # Convert word to tensor of indices
        word_indices = [self.char_to_idx.get(char, 0) for char in word]
        word_tensor = torch.tensor(word_indices, dtype=torch.long)

        # Convert phonemes to tensor of indices
        phoneme_indices = [self.phoneme_to_idx.get(phoneme, 0) for phoneme in phonemes]
        phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long)

        return word_tensor, phoneme_tensor

# Dataset that returns (inputs, targets) - Format 1
class SpanishPhoneticDatasetFormat1(BaseSpanishPhoneticDataset):
    def __getitem__(self, idx):
        word_tensor, phoneme_tensor = self._get_tensors(idx)
        return word_tensor, phoneme_tensor

# Dataset that returns (inputs, targets, mask) - Format 2
class SpanishPhoneticDatasetFormat2(BaseSpanishPhoneticDataset):
    def __getitem__(self, idx):
        word_tensor, phoneme_tensor = self._get_tensors(idx)
        # Create a mask for the target (all True in this case)
        mask = torch.ones_like(phoneme_tensor, dtype=torch.bool)
        return word_tensor, phoneme_tensor, mask

# Dataset that returns (inputs, targets, input_mask, target_mask) - Format 3
class SpanishPhoneticDatasetFormat3(BaseSpanishPhoneticDataset):
    def __getitem__(self, idx):
        word_tensor, phoneme_tensor = self._get_tensors(idx)
        # Create masks for both input and target (all True in this case)
        input_mask = torch.ones_like(word_tensor, dtype=torch.bool)
        target_mask = torch.ones_like(phoneme_tensor, dtype=torch.bool)
        return word_tensor, phoneme_tensor, input_mask, target_mask

# For backward compatibility
SpanishPhoneticDataset = SpanishPhoneticDatasetFormat2

# Collate function for Format 1: (inputs, targets)
def collate_fn_format1(batch):
    # Sort the batch by word length (descending)
    batch = list(batch)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    # Get the data
    words, phonemes = zip(*batch)

    # Pad the sequences
    words_padded = nn.utils.rnn.pad_sequence(words, batch_first=True)
    phonemes_padded = nn.utils.rnn.pad_sequence(phonemes, batch_first=True)

    return words_padded, phonemes_padded

# Collate function for Format 2: (inputs, targets, mask)
def collate_fn_format2(batch):
    # Sort the batch by word length (descending)
    batch = list(batch)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    # Get the data
    words, phonemes, masks = zip(*batch)

    # Pad the sequences
    words_padded = nn.utils.rnn.pad_sequence(words, batch_first=True)
    phonemes_padded = nn.utils.rnn.pad_sequence(phonemes, batch_first=True)
    masks_padded = nn.utils.rnn.pad_sequence(masks, batch_first=True)

    return words_padded, phonemes_padded, masks_padded

# Collate function for Format 3: (inputs, targets, input_mask, target_mask)
def collate_fn_format3(batch):
    # Sort the batch by word length (descending)
    batch = list(batch)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    # Get the data
    words, phonemes, input_masks, target_masks = zip(*batch)

    # Pad the sequences
    words_padded = nn.utils.rnn.pad_sequence(words, batch_first=True)
    phonemes_padded = nn.utils.rnn.pad_sequence(phonemes, batch_first=True)
    input_masks_padded = nn.utils.rnn.pad_sequence(input_masks, batch_first=True)
    target_masks_padded = nn.utils.rnn.pad_sequence(target_masks, batch_first=True)

    return words_padded, phonemes_padded, input_masks_padded, target_masks_padded

# For backward compatibility
collate_fn = collate_fn_format2

# Custom loss function
def phonetic_loss_fn(outputs, targets, target_mask, input_mask=None):
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

    # Ensure target_mask is a tensor or convert it to one
    if not isinstance(target_mask, torch.Tensor):
        if isinstance(target_mask, tuple) and len(target_mask) > 0:
            target_mask = target_mask[0]  # Take the first element if it's a tuple
            if not isinstance(target_mask, torch.Tensor):
                if isinstance(target_mask, bool):
                    # Create a tensor with all True values with the same shape as targets
                    target_mask = torch.ones_like(targets, dtype=torch.bool)
                else:
                    raise TypeError(f"Expected target_mask[0] to be a tensor or bool, got {type(target_mask)}")
        elif isinstance(target_mask, bool):
            # Create a tensor with all True values with the same shape as targets
            target_mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            raise TypeError(f"Expected target_mask to be a tensor or bool, got {type(target_mask)}")

    # input_mask is optional and not used in this loss function
    # but we include it for compatibility with the encoder's calculate_code_length method

    # Get the dimensions
    batch_size, seq_len, output_size = outputs.shape

    # If the batch sizes don't match, we need to handle this case
    if batch_size != targets.size(0) or batch_size != target_mask.size(0):
        # If outputs has batch size 1, we need to repeat it to match the batch size of targets and mask
        if batch_size == 1:
            outputs = outputs.repeat(targets.size(0), 1, 1)
            batch_size = outputs.shape[0]
        # If targets has batch size 1, we need to repeat it to match the batch size of outputs
        elif targets.size(0) == 1:
            targets = targets.repeat(batch_size, 1)
        # If mask has batch size 1, we need to repeat it to match the batch size of outputs
        elif target_mask.size(0) == 1:
            target_mask = target_mask.repeat(batch_size, 1)
        else:
            # If the batch sizes are different and neither is 1, we can't handle this case
            raise ValueError(f"Batch size mismatch: outputs {batch_size}, targets {targets.size(0)}, target_mask {target_mask.size(0)}")

    # Make sure targets and mask have the right shape
    if targets.dim() == 2:
        targets_seq_len = targets.size(1)
    else:
        targets_seq_len = 1

    if target_mask.dim() == 2:
        mask_seq_len = target_mask.size(1)
    else:
        mask_seq_len = 1

    # Ensure all sequences have the same length by padding or truncating
    min_seq_len = min(seq_len, targets_seq_len, mask_seq_len)

    # Truncate if necessary
    outputs = outputs[:, :min_seq_len, :]
    if targets.dim() == 2:
        targets = targets[:, :min_seq_len]
    if target_mask.dim() == 2:
        target_mask = target_mask[:, :min_seq_len]

    # Reshape outputs to [batch_size * min_seq_len, output_size]
    outputs = outputs.reshape(-1, output_size)

    # Reshape targets to [batch_size * min_seq_len]
    targets = targets.reshape(-1)

    # Reshape mask to [batch_size * min_seq_len]
    target_mask = target_mask.reshape(-1)

    # Apply the mask
    masked_outputs = outputs[target_mask]
    masked_targets = targets[target_mask]

    # Check if target indices are within bounds
    output_size = masked_outputs.size(-1)
    if torch.any(masked_targets >= output_size):
        raise ValueError(f"Error: Target indices out of bounds. Target max: {masked_targets.max().item()}, Output size: {output_size}. The model's output size is too small for the dataset.")

    # Calculate cross entropy loss
    return F.cross_entropy(masked_outputs, masked_targets, reduction='none')

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Path to the Spanish phonetic data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")

    # Test all three dataset formats
    test_format1(data_path)
    test_format2(data_path)
    test_format3(data_path)

def test_format1(data_path):
    """Test encoders with Format 1: (inputs, targets)"""
    print("\n" + "="*80)
    print("TESTING FORMAT 1: (inputs, targets)")
    print("="*80)

    # Create the dataset
    dataset = SpanishPhoneticDatasetFormat1(data_path, max_samples=500)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(dataset.phoneme_to_idx)}")

    # Test BlockEncoder
    print("\nTesting BlockEncoder with Format 1...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(dataset.phoneme_to_idx)
        }
    )
    block_encoder = BlockEncoder(
        model_class=model_class,
        loss_fn=phonetic_loss_fn
    )

    # Encode with BlockEncoder (one-shot approach)
    model, code_length, code_length_history = block_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (Block, Format 1)",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        stop_points=[0.5, 1.0],
        patience=5,
        collate_fn=collate_fn_format1,
        use_device_handling=False
    )

    print(f"Block Encoder (Format 1) - Code length: {code_length}.")

    # Test MIREncoder
    print("\nTesting MIREncoder with Format 1...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
        loss_fn=phonetic_loss_fn
    )

    # Encode with MIREncoder (one-shot approach)
    model, code_length, code_length_history, ema_params, beta, replay_streams = mir_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (MIR, Format 1)",
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format1,
        use_device_handling=False,
        use_beta=True,
        use_ema=True
    )

    print(f"MIR Encoder (Format 1) - Code length: {code_length}.")

def test_format2(data_path):
    """Test encoders with Format 2: (inputs, targets, mask)"""
    print("\n" + "="*80)
    print("TESTING FORMAT 2: (inputs, targets, mask)")
    print("="*80)

    # Create the dataset
    dataset = SpanishPhoneticDatasetFormat2(data_path, max_samples=500)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(dataset.phoneme_to_idx)}")

    # Test BlockEncoder
    print("\nTesting BlockEncoder with Format 2...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(dataset.phoneme_to_idx)
        }
    )
    block_encoder = BlockEncoder(
        model_class=model_class,
        loss_fn=phonetic_loss_fn
    )

    # Test staged approach (initialize, step, finalize)
    print("\nTesting BlockEncoder with staged approach (initialize, step, finalize)...")
    # Initialize
    state, train_chunks, eval_chunks, batch_size, shuffle, collate_fn_result = block_encoder.initialize(
        dataset=dataset,
        stop_points=[0.5, 1.0],
        batch_size=32,
        learning_rate=0.001,
        seed=42,
        shuffle=True,
        collate_fn=collate_fn_format2
    )

    # Create dataloader for the first chunk
    train_loader = DataLoader(train_chunks[0], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_format2)

    # Step through batches
    for batch in train_loader:
        block_encoder.step(state, batch)

    # Evaluate on the first chunk
    eval_loader = DataLoader(eval_chunks[0], batch_size=batch_size, shuffle=False, collate_fn=collate_fn_format2)
    block_encoder.eval_code_length(state, eval_loader)

    print(f"Block Encoder (staged approach, Format 2) - Code length: {state.code_length}.")

    # Encode with BlockEncoder (one-shot approach)
    model, code_length, code_length_history = block_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (Block, Format 2)",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        stop_points=[0.5, 1.0],
        patience=5,
        collate_fn=collate_fn_format2,
        use_device_handling=False,
        
    )

    print(f"Block Encoder (Format 2) - Code length: {code_length}.")

    # Test MIREncoder
    print("\nTesting MIREncoder with Format 2...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
        loss_fn=phonetic_loss_fn
    )

    # Encode with MIREncoder (one-shot approach)
    model, code_length, code_length_history, ema_params, beta, replay_streams = mir_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (MIR, Format 2)",
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format2,
        use_device_handling=False,
        use_beta=True,
        use_ema=True
    )

    print(f"MIR Encoder (Format 2) - Code length: {code_length}.")

def test_format3(data_path):
    """Test encoders with Format 3: (inputs, targets, input_mask, target_mask)"""
    print("\n" + "="*80)
    print("TESTING FORMAT 3: (inputs, targets, input_mask, target_mask)")
    print("="*80)

    # Create the dataset
    dataset = SpanishPhoneticDatasetFormat3(data_path, max_samples=500)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(dataset.phoneme_to_idx)}")

    # Test BlockEncoder
    print("\nTesting BlockEncoder with Format 3...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(dataset.phoneme_to_idx)
        }
    )
    block_encoder = BlockEncoder(
        model_class=model_class,
        loss_fn=phonetic_loss_fn
    )

    # Encode with BlockEncoder (one-shot approach)
    model, code_length, code_length_history = block_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (Block, Format 3)",
        epochs=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        stop_points=[0.5, 1.0],
        patience=5,
        collate_fn=collate_fn_format3,
        use_device_handling=False
    )

    print(f"Block Encoder (Format 3) - Code length: {code_length}.")

    # Test MIREncoder
    print("\nTesting MIREncoder with Format 3...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
        loss_fn=phonetic_loss_fn
    )

    # Encode with MIREncoder (one-shot approach)
    model, code_length, code_length_history, ema_params, beta, replay_streams = mir_encoder.encode(
        dataset=dataset,
        set_name="Spanish Phonetic (MIR, Format 3)",
        n_replay_streams=2,
        learning_rate=0.001,
        batch_size=32,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format3,
        use_device_handling=False,
        use_beta=True,
        use_ema=True
    )

    print(f"MIR Encoder (Format 3) - Code length: {code_length}.")

if __name__ == "__main__":
    main()
