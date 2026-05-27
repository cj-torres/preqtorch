

def test_preq_loader_from_indexables():
    xs = [torch.randn(4) for _ in range(5)]
    ys = [torch.tensor(i % 2) for i in range(5)]
    ms = [torch.tensor(True) for _ in range(5)]

    loader = PrequentialDataLoader(inputs=xs, targets=ys, masks=ms, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert hasattr(batch, "inputs")
    assert hasattr(batch, "targets")
    assert hasattr(batch, "output_mask")
    assert hasattr(batch, "target_mask")





def test_encoder_result_fields():
    r = EncoderResult(model='m', code_length=1.0, history=[1,2,3])
    assert r.model == 'm'
    assert r.code_length == 1.0
    assert r.history == [1,2,3]

    r2 = EncoderResult(model='m', code_length=1.0, history=[], ema_params={}, beta='b', replay='r')
    assert r2.replay == 'r'

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader

# Import directly from the package
from preqtorch import BlockEncoder, MIREncoder, ModelClass, EncoderResult, PrequentialDataLoader, PrequentialDataset

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
        word_indices = [1]+[self.char_to_idx.get(char, 0) for char in word]
        word_tensor = torch.tensor(word_indices, dtype=torch.long)

        # Convert phonemes to tensor of indices
        phoneme_indices = [self.phoneme_to_idx.get(phoneme, 0) for phoneme in phonemes]+[1]
        phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long)

        return word_tensor, phoneme_tensor

# Dataset builders backed by PrequentialDataset
def build_spanish_dataset_format1(base_dataset):
    inputs, targets = zip(*[base_dataset._get_tensors(i) for i in range(len(base_dataset))])
    return PrequentialDataset(inputs=inputs, targets=targets)


def build_spanish_dataset_format2(base_dataset):
    samples = [base_dataset._get_tensors(i) for i in range(len(base_dataset))]
    inputs, targets = zip(*samples)
    masks = [torch.ones_like(t, dtype=torch.bool) for t in targets]
    return PrequentialDataset(inputs=inputs, targets=targets, masks=masks)


def build_spanish_dataset_format3(base_dataset):
    samples = [base_dataset._get_tensors(i) for i in range(len(base_dataset))]
    inputs, targets = zip(*samples)
    output_masks = [torch.ones_like(t, dtype=torch.bool) for t in targets]
    target_masks = [torch.ones_like(t, dtype=torch.bool) for t in targets]
    return PrequentialDataset(inputs=inputs, targets=targets, masks=output_masks, target_masks=target_masks)

# For backward compatibility
SpanishPhoneticDataset = build_spanish_dataset_format2

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

    # Ensure both tensors have the same size
    max_len = max(words_padded.size(1), phonemes_padded.size(1))

    # Pad words if needed
    if words_padded.size(1) < max_len:
        padding = torch.zeros(words_padded.size(0), max_len - words_padded.size(1), dtype=words_padded.dtype, device=words_padded.device)
        words_padded = torch.cat([words_padded, padding], dim=1)

    # Pad phonemes if needed
    if phonemes_padded.size(1) < max_len:
        padding = torch.zeros(phonemes_padded.size(0), max_len - phonemes_padded.size(1), dtype=phonemes_padded.dtype, device=phonemes_padded.device)
        phonemes_padded = torch.cat([phonemes_padded, padding], dim=1)

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

    # Ensure both tensors have the same size
    max_len = max(words_padded.size(1), phonemes_padded.size(1))

    # Pad words if needed
    if words_padded.size(1) < max_len:
        padding = torch.zeros(words_padded.size(0), max_len - words_padded.size(1), dtype=words_padded.dtype, device=words_padded.device)
        words_padded = torch.cat([words_padded, padding], dim=1)

    # Pad phonemes if needed
    if phonemes_padded.size(1) < max_len:
        padding = torch.zeros(phonemes_padded.size(0), max_len - phonemes_padded.size(1), dtype=phonemes_padded.dtype, device=phonemes_padded.device)
        phonemes_padded = torch.cat([phonemes_padded, padding], dim=1)

    # Also pad the masks to match phonemes
    if masks_padded.size(1) < max_len:
        mask_padding = torch.zeros(masks_padded.size(0), max_len - masks_padded.size(1), dtype=masks_padded.dtype, device=masks_padded.device)
        masks_padded = torch.cat([masks_padded, mask_padding], dim=1)

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
    output_masks_padded = nn.utils.rnn.pad_sequence(input_masks, batch_first=True)
    target_masks_padded = nn.utils.rnn.pad_sequence(target_masks, batch_first=True)

    # Ensure both tensors have the same size
    max_len = max(words_padded.size(1), phonemes_padded.size(1))

    # Pad words if needed
    if words_padded.size(1) < max_len:
        padding = torch.zeros(words_padded.size(0), max_len - words_padded.size(1), dtype=words_padded.dtype, device=words_padded.device)
        words_padded = torch.cat([words_padded, padding], dim=1)

    # Pad phonemes if needed
    if phonemes_padded.size(1) < max_len:
        padding = torch.zeros(phonemes_padded.size(0), max_len - phonemes_padded.size(1), dtype=phonemes_padded.dtype, device=phonemes_padded.device)
        phonemes_padded = torch.cat([phonemes_padded, padding], dim=1)

    # Also pad the target masks to match phonemes
    if target_masks_padded.size(1) < max_len:
        mask_padding = torch.zeros(target_masks_padded.size(0), max_len - target_masks_padded.size(1), dtype=target_masks_padded.dtype, device=target_masks_padded.device)
        target_masks_padded = torch.cat([target_masks_padded, mask_padding], dim=1)

    # Also pad the input masks to match words
    if output_masks_padded.size(1) < max_len:
        mask_padding = torch.zeros(output_masks_padded.size(0), max_len - output_masks_padded.size(1), dtype=output_masks_padded.dtype, device=output_masks_padded.device)
        output_masks_padded = torch.cat([output_masks_padded, mask_padding], dim=1)


    return words_padded, phonemes_padded, output_masks_padded, target_masks_padded

# For backward compatibility
collate_fn = collate_fn_format2

# Custom loss function
def phonetic_loss_fn(outputs, targets, output_mask, target_mask):
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

    # Apply masks to outputs and targets
    masked_outputs = outputs[output_mask]
    masked_targets = targets[target_mask]

    return F.cross_entropy(masked_outputs, masked_targets, reduction='none')

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test all three dataset formats
    test_format1()
    test_format2()
    test_format3()

    # Test default encoding function
    test_default_encoding_fn()

    # Test custom encoding function passed directly to encode()
    test_custom_encoding_fn_in_encode()

    # Test MIREncoder with different beta and EMA configurations
    test_mir_encoder_without_beta()
    test_mir_encoder_without_ema()
    test_mir_encoder_without_both()

def test_format1():
    """Test encoders with Format 1: (inputs, targets)"""
    # Define data_path inside the test function
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")
    print("\n" + "="*80)
    print("TESTING FORMAT 1: (inputs, targets)")
    print("="*80)

    # Create the dataset
    base_dataset = BaseSpanishPhoneticDataset(data_path, max_samples=500)
    dataset = build_spanish_dataset_format1(base_dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(base_dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(base_dataset.phoneme_to_idx)}")

    # Test BlockEncoder
    print("\nTesting BlockEncoder with Format 1...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    block_encoder = BlockEncoder(
        model_class=model_class,
    )

    # Encode with BlockEncoder (one-shot approach)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_format1)
    result = block_encoder.encode(
        train_dataloader=[loader],
        eval_dataloaders=[loader],
        set_name="Spanish Phonetic (Block, Format 1)",
        epochs=2,
        learning_rate=0.001,
        seed=42,
        patience=5,
        collate_fn=collate_fn_format1,
        use_device_handling=False
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"Block Encoder (Format 1) - Code length: {code_length}.")

    # Test MIREncoder
    print("\nTesting MIREncoder with Format 1...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
    )

    # Encode with MIREncoder (one-shot approach)
    result = mir_encoder.encode(
        dataloader=loader,
        set_name="Spanish Phonetic (MIR, Format 1)",
        n_replay_samples=2,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format1,
        use_device_handling=False,
        use_beta=True,
        use_ema=True
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    ema_params, beta, replay_streams = result.ema_params, result.beta, result.replay
    print(f"MIR Encoder (Format 1) - Code length: {code_length}.")

def test_format2():
    """Test encoders with Format 2: (inputs, targets, mask)"""
    # Define data_path inside the test function
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")
    print("\n" + "="*80)
    print("TESTING FORMAT 2: (inputs, targets, mask)")
    print("="*80)

    # Create the dataset
    base_dataset = BaseSpanishPhoneticDataset(data_path, max_samples=500)
    dataset = build_spanish_dataset_format2(base_dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(base_dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(base_dataset.phoneme_to_idx)}")

    # Test BlockEncoder
    print("\nTesting BlockEncoder with Format 2...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    block_encoder = BlockEncoder(
        model_class=model_class,
    )

    # Encode with BlockEncoder (one-shot approach)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_format1)
    result = block_encoder.encode(
        train_dataloader=[loader],
        eval_dataloaders=[loader],
        set_name="Spanish Phonetic (Block, Format 2)",
        epochs=2,
        learning_rate=0.001,
        seed=42,
        patience=5,
        collate_fn=collate_fn_format2,
        use_device_handling=False,

    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"Block Encoder (Format 2) - Code length: {code_length}.")

    # Test MIREncoder
    print("\nTesting MIREncoder with Format 2...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
    )

    # Encode with MIREncoder (one-shot approach)
    result = mir_encoder.encode(
        dataloader=loader,
        set_name="Spanish Phonetic (MIR, Format 2)",
        n_replay_samples=2,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format2,
        use_device_handling=False,
        use_beta=True,
        use_ema=True
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    ema_params, beta, replay_streams = result.ema_params, result.beta, result.replay
    print(f"MIR Encoder (Format 2) - Code length: {code_length}.")

def test_format3():
    """Test encoders with Format 3: (inputs, targets, input_mask, target_mask)"""
    # Define data_path inside the test function
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")
    print("\n" + "="*80)
    print("TESTING FORMAT 3: (inputs, targets, input_mask, target_mask)")
    print("="*80)

    # Create the dataset
    base_dataset = BaseSpanishPhoneticDataset(data_path, max_samples=500)
    dataset = build_spanish_dataset_format3(base_dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(base_dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(base_dataset.phoneme_to_idx)}")

    # Test BlockEncoder
    print("\nTesting BlockEncoder with Format 3...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    block_encoder = BlockEncoder(
        model_class=model_class,
    )

    # Encode with BlockEncoder (one-shot approach)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_format1)
    result = block_encoder.encode(
        train_dataloader=[loader],
        eval_dataloaders=[loader],
        set_name="Spanish Phonetic (Block, Format 3)",
        epochs=2,
        learning_rate=0.001,
        seed=42,
        patience=5,
        collate_fn=collate_fn_format3,
        use_device_handling=False
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"Block Encoder (Format 3) - Code length: {code_length}.")

    # Test MIREncoder
    print("\nTesting MIREncoder with Format 3...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
    )

    # Encode with MIREncoder (one-shot approach)
    result = mir_encoder.encode(
        dataloader=loader,
        set_name="Spanish Phonetic (MIR, Format 3)",
        n_replay_samples=2,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format3,
        use_device_handling=False,
        use_beta=True,
        use_ema=True
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    ema_params, beta, replay_streams = result.ema_params, result.beta, result.replay
    print(f"MIR Encoder (Format 3) - Code length: {code_length}.")

def test_default_encoding_fn():
    """Test encoders with the default encoding function"""
    # Define data_path inside the test function
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")
    print("\n" + "="*80)
    print("TESTING DEFAULT ENCODING FUNCTION")
    print("="*80)

    # Create the dataset
    base_dataset = BaseSpanishPhoneticDataset(data_path, max_samples=500)
    dataset = build_spanish_dataset_format3(base_dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(base_dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(base_dataset.phoneme_to_idx)}")

    # Test BlockEncoder with default encoding function
    print("\nTesting BlockEncoder with default encoding function...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    block_encoder = BlockEncoder(
        model_class=model_class
    )

    # Encode with BlockEncoder (one-shot approach)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_format1)
    result = block_encoder.encode(
        train_dataloader=[loader],
        eval_dataloaders=[loader],
        set_name="Spanish Phonetic (Block, Default Encoding)",
        epochs=2,
        learning_rate=0.001,
        seed=42,
        patience=5,
        collate_fn=collate_fn_format3,
        use_device_handling=False
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"Block Encoder (Default Encoding) - Code length: {code_length}.")

    # Test MIREncoder with default encoding function
    print("\nTesting MIREncoder with default encoding function...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class
    )

    # Encode with MIREncoder (one-shot approach)
    result = mir_encoder.encode(
        dataloader=loader,
        set_name="Spanish Phonetic (MIR, Default Encoding)",
        n_replay_samples=2,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format3,
        use_device_handling=False,
        use_beta=True,
        use_ema=True
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"MIR Encoder (Default Encoding) - Code length: {code_length}.")

def test_custom_encoding_fn_in_encode():
    """Test encoders with custom encoding function passed directly to encode()"""
    # Define data_path inside the test function
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")
    print("\n" + "="*80)
    print("TESTING CUSTOM ENCODING FUNCTION PASSED TO ENCODE()")
    print("="*80)

    # Create the dataset
    base_dataset = BaseSpanishPhoneticDataset(data_path, max_samples=500)
    dataset = build_spanish_dataset_format3(base_dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(base_dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(base_dataset.phoneme_to_idx)}")

    # Define a custom encoding function with a multiplier to make it different from the default
    def custom_encoding_fn(outputs, targets, output_mask, target_mask):
        # Apply masks to outputs and targets
        masked_outputs = outputs[output_mask]
        masked_targets = targets[target_mask]
        # Use a multiplier of 1.5 to make it different from the default
        return 1.5 * F.cross_entropy(masked_outputs, masked_targets, reduction='none')/torch.log(torch.tensor(2.0, device=outputs.device))

    # Test BlockEncoder with custom encoding function passed to encode()
    print("\nTesting BlockEncoder with custom encoding function passed to encode()...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    block_encoder = BlockEncoder(
        model_class=model_class
    )

    # Encode with BlockEncoder (one-shot approach) with custom encoding function
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_format3)
    result = block_encoder.encode(
        train_dataloader=[loader],
        eval_dataloaders=[loader],
        set_name="Spanish Phonetic (Block, Custom Encoding)",
        epochs=2,
        learning_rate=0.001,
        seed=42,
        patience=5,
        collate_fn=collate_fn_format3,
        use_device_handling=False,
        encoding_fn=custom_encoding_fn  # Pass custom encoding function here
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"Block Encoder (Custom Encoding) - Code length: {code_length}.")

    # Test MIREncoder with custom encoding function passed to encode()
    print("\nTesting MIREncoder with custom encoding function passed to encode()...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class
    )

    # Encode with MIREncoder (one-shot approach) with custom encoding function
    result = mir_encoder.encode(
        dataloader=loader,
        set_name="Spanish Phonetic (MIR, Custom Encoding)",
        n_replay_samples=2,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format3,
        use_device_handling=False,
        use_beta=True,
        use_ema=True,
        encoding_fn=custom_encoding_fn  # Pass custom encoding function here
    )

    print(f"MIR Encoder (Custom Encoding) - Code length: {code_length}.")

def test_mir_encoder_without_beta():
    """Test MIREncoder without beta (use_beta=False, use_ema=True)"""
    # Define data_path inside the test function
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")
    print("\n" + "="*80)
    print("TESTING MIR ENCODER WITHOUT BETA (use_beta=False, use_ema=True)")
    print("="*80)

    # Create the dataset
    base_dataset = BaseSpanishPhoneticDataset(data_path, max_samples=500)
    dataset = build_spanish_dataset_format3(base_dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(base_dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(base_dataset.phoneme_to_idx)}")

    # Test MIREncoder without beta
    print("\nTesting MIREncoder without beta...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
    )

    # Encode with MIREncoder (one-shot approach) without beta
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_format3)
    result = mir_encoder.encode(
        dataloader=loader,
        set_name="Spanish Phonetic (MIR, Without Beta)",
        n_replay_samples=2,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format3,
        use_device_handling=False,
        use_beta=False,  # Disable beta
        use_ema=True     # Keep EMA enabled
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"MIR Encoder (Without Beta) - Code length: {code_length}.")

def test_mir_encoder_without_ema():
    """Test MIREncoder without EMA (use_beta=True, use_ema=False)"""
    # Define data_path inside the test function
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")
    print("\n" + "="*80)
    print("TESTING MIR ENCODER WITHOUT EMA (use_beta=True, use_ema=False)")
    print("="*80)

    # Create the dataset
    base_dataset = BaseSpanishPhoneticDataset(data_path, max_samples=500)
    dataset = build_spanish_dataset_format3(base_dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(base_dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(base_dataset.phoneme_to_idx)}")

    # Test MIREncoder without EMA
    print("\nTesting MIREncoder without EMA...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
    )

    # Encode with MIREncoder (one-shot approach) without EMA
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_format3)
    result = mir_encoder.encode(
        dataloader=loader,
        set_name="Spanish Phonetic (MIR, Without EMA)",
        n_replay_samples=2,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format3,
        use_device_handling=False,
        use_beta=True,   # Keep beta enabled
        use_ema=False    # Disable EMA
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"MIR Encoder (Without EMA) - Code length: {code_length}.")

def test_mir_encoder_without_both():
    """Test MIREncoder without both beta and EMA (use_beta=False, use_ema=False)"""
    # Define data_path inside the test function
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spa_latn_la_broad.tsv")
    print("\n" + "="*80)
    print("TESTING MIR ENCODER WITHOUT BOTH BETA AND EMA (use_beta=False, use_ema=False)")
    print("="*80)

    # Create the dataset
    base_dataset = BaseSpanishPhoneticDataset(data_path, max_samples=500)
    dataset = build_spanish_dataset_format3(base_dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {len(base_dataset.char_to_idx)}")
    print(f"Number of phonemes: {len(base_dataset.phoneme_to_idx)}")

    # Test MIREncoder without both beta and EMA
    print("\nTesting MIREncoder without both beta and EMA...")
    model_class = ModelClass(
        model=SimplePhoneticModel,
        device='cpu',
        kwargs={
            'input_size': len(base_dataset.char_to_idx),
            'hidden_size': 64,
            'output_size': len(base_dataset.phoneme_to_idx)
        }
    )
    mir_encoder = MIREncoder(
        model_class=model_class,
    )

    # Encode with MIREncoder (one-shot approach) without both beta and EMA
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_format3)
    result = mir_encoder.encode(
        dataloader=loader,
        set_name="Spanish Phonetic (MIR, Without Both)",
        n_replay_samples=2,
        learning_rate=0.001,
        seed=42,
        alpha=0.1,
        collate_fn=collate_fn_format3,
        use_device_handling=False,
        use_beta=False,  # Disable beta
        use_ema=False    # Disable EMA
    )

    model, code_length, code_length_history = result.model, result.code_length, result.history
    print(f"MIR Encoder (Without Both Beta and EMA) - Code length: {code_length}.")

if __name__ == "__main__":
    main()
