# PreqTorch

A PyTorch-based library for calculating the prequential codelength of datasets. This toolkit allows for calculating the stochastic complexity of a dataset given it and a model class.

## Overview

PreqTorch provides tools for prequential encoding in PyTorch. Prequential encoding is a technique for evaluating datasets in an online learning setting, where the model is updated after each prediction.

The library includes:
- Prequential encoders (BlockEncoder, MIREncoder)

## Installation

### From PyPI

```bash
pip install preqtorch
```

### From Source

```bash
git clone https://github.com/cj-torres/preqtorch.git
cd preqtorch
pip install -e .
```

## Requirements

PreqTorch has the following requirements:
- Python 3.6+
- PyTorch 1.7+
- NumPy


# Usage

I built this package to stop myself from rewriting the same prequential encoding process over and over. For this reason, the package wraps your dataset, model, and if necessary, a collate function and encoding (loss) function.

The rest of the document will review the required formats for each of these. In my first iteration I've tried to strike a balance between flexibility and brevity.

### Dataset formatting

For PreqTorch to work properly, your datasets must:

1. Be organized as tuples of tensors or tuples of tuples including tensors
2. Return data in one of the following formats:
   - `(inputs, targets)` - Basic format without masks
   - `(inputs, targets, mask)` - Format with a shared mask for both model outputs and targets
   - `(inputs, targets, output_mask, target_mask)` - Format with separate masks for outputs and targets
3. Be compatible with PyTorch's Dataset class

### Collate Function

When using PreqTorch encoders, you may provide your own collate function at creation time. This function should:

- Take a batch of samples and combine them into a single batch
- Return data in one of the supported formats:
  - `(inputs, targets)`
  - `(inputs, targets, mask)` - shared mask for both model outputs and targets
  - `(inputs, targets, output_mask, target_mask)` - separate masks for outputs and targets
- Handle any specific requirements of your dataset

Examples of collate functions for different dataset formats:

```python
# Basic collate function (inputs, targets)
def basic_collate_fn(batch):
    # Unpack the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack inputs and targets into tensors
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return inputs, targets

# Collate function with shared mask (inputs, targets, mask)
def masked_collate_fn(batch):
    # Unpack the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Create or extract masks (example: mask based on non-zero values)
    masks = [torch.ones_like(item[1], dtype=torch.bool) for item in batch]

    # Stack inputs, targets, and masks into tensors
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    masks = torch.stack(masks)

    return inputs, targets, masks

# Collate function with separate masks (inputs, targets, output_mask, target_mask)
def separate_masks_collate_fn(batch):
    # Unpack the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Create or extract masks (example: different masks for outputs and targets)
    # Note: output_mask will be applied to model outputs, which should have the same shape as inputs
    output_masks = [torch.ones_like(item[0], dtype=torch.bool) for item in batch]
    target_masks = [torch.ones_like(item[1], dtype=torch.bool) for item in batch]

    # Stack inputs, targets, and masks into tensors
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    output_masks = torch.stack(output_masks)
    target_masks = torch.stack(target_masks)

    return inputs, targets, output_masks, target_masks
```

### Encoding Function

By default encoders will attempt to use cross entropy loss, returning code lengths calculated from the loss in units of bits. However, a custom encoding function may be supplied. No matter what function is supplied, it will be called like this:

```python
code_lengths = encoding_fn(outputs, target, output_mask, target_mask)
```

You can write the function however you wish! But understand that this is the call that will be made internally.


## Encoders

The package supports two types of prequential encoders, themselves approximations of true prequential encoding (which is unwieldy).

### Block Encoding

Block encoding divides the dataset into blocks and trains the model on each block sequentially. See Blier, et al. (2018) for details.

```python
import torch
from preqtorch import BlockEncoder

# Define a model class
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Create a block encoder
encoder = BlockEncoder(
    model_class=MyModel,
    loss_fn=torch.nn.functional.cross_entropy
)

# Encode a dataset using block encoding
model, code_length, history = encoder.encode(
    dataset=my_dataset,
    set_name="My Dataset",
    stop_points=[0.125, 0.25, 0.5, 1.0],  # Points (in proportion) to stop and evaluate
    batch_size=32,
    seed=42,
    learning_rate=0.001,
    epochs=50,
    patience=20,
    collate_fn=my_collate_fn  # Your custom collate function
)
```

### MIR Encoding

MIR (Mini-batch Incremental/Replay) encoding uses replay buffers or streams to revisit previous data. See Bornschein, et al. (2022) for details.

```python
from preqtorch import MIREncoder

# Create a MIR encoder
encoder = MIREncoder(
    model_class=MyModel,
    loss_fn=torch.nn.functional.cross_entropy
)

# Encode a dataset using MIR encoding
model, code_length, history, ema_params, beta, replay = encoder.encode(
    dataset=my_dataset,
    set_name="My Dataset",
    n_replay_samples=2,  # Number of replay streams or buffer size
    learning_rate=0.001,
    batch_size=32,
    seed=42,
    alpha=0.1,  # EMA update rate
    collate_fn=my_collate_fn,  # Your custom collate function
    use_beta=True,  # Whether to use learnable temperature parameter
    use_ema=True,  # Whether to use exponential moving average
    replay_type="buffer"  # Type of replay: "buffer" or "streams"
)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## See also

Bornschein, J., Li, Y., & Hutter, M. (2022). Sequential learning of neural networks for prequential mdl. arXiv preprint arXiv:2210.07931.

Blier, L., & Ollivier, Y. (2018). The description length of deep learning models. Advances in Neural Information Processing Systems, 31.