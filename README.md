# PreqTorch

A PyTorch-based library for calculating the prequential codelength of datasets. This toolkit allows for calculating the stochastic complexity of a dataset given a dataset and model class.

## Overview

PreqTorch provides tools for prequential encoding in PyTorch. Prequential encoding is a technique for evaluating datasets in an online learning setting, where the model is updated after each prediction.

The library includes:
- Prequential encoders (`BlockEncoder`, `MIREncoder`)
- Model wrapper (`ModelClass`) for explicit initialization/device behavior
- Structured return object (`EncoderResult`) with named fields
- Canonical batch object (`PrequentialBatch`), dataset (`PrequentialDataset`), and prequential-native loader (`PrequentialDataLoader`)

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

## Usage

PreqTorch is designed around the idea that **model initialization is part of the model definition**. You pass a `ModelClass` wrapper that can sample freshly initialized models during encoding.

You may provide custom:
- dataset format
- `collate_fn`
- encoding function (`encoding_fn`)

directly to `encode(...)`, with the user owning dataloader construction.

### Dataset formatting


### PrequentialDataLoader

`PrequentialDataset` and `PrequentialDataLoader` follow PyTorch conventions: dataset owns indexing logic; dataloader owns batching/shuffling/iteration.

It accepts indexable sources at initialization and requires keyword arguments `inputs` and `targets`. Optional `masks` and `target_masks` are also supported.

```python
from preqtorch import PrequentialDataset, PrequentialDataLoader

dataset = PrequentialDataset(
    inputs=my_inputs,
    targets=my_targets,
    masks=my_output_masks,          # optional
    target_masks=my_target_masks,   # optional
)

loader = PrequentialDataLoader(
    inputs=my_inputs,
    targets=my_targets,
    masks=my_output_masks,
    target_masks=my_target_masks,
    shuffle=True,
)
```

The loader validates that indexable sources line up in length and that indexed values are tensors (or tuples of tensors), then yields canonical `PrequentialBatch` objects.


> Note: The library now uses a canonical `PrequentialBatch` internally. If you do not pass a custom `collate_fn`, encoders default to `prequential_collate`.

For PreqTorch to work properly, datasets should return one of these formats:

1. `(inputs, targets)`
2. `(inputs, targets, mask)` where the mask is shared between outputs and targets
3. `(inputs, targets, output_mask, target_mask)`

These formats can come directly from your dataset, or from your custom collate function.

### Collate function

When using PreqTorch encoders, your `collate_fn` should combine a list of samples into one of the supported batch formats above.

### Encoding function contract

By default, encoders use a cross-entropy based code-length function (in bits). You can supply a custom one. It will be called as:

```python
code_lengths = encoding_fn(outputs, targets, output_mask, target_mask)
```

### Block encoding

```python
import torch
from preqtorch import BlockEncoder, ModelClass

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

model_class = ModelClass(MyModel, device="cpu")
encoder = BlockEncoder(model_class=model_class)

result = encoder.encode(
    train_dataloader=[train_loader_1, train_loader_2],
    eval_dataloaders=[eval_loader_1, eval_loader_2],
    set_name="My Dataset",
    seed=42,
    learning_rate=0.001,
    epochs=50,
    patience=20,
    collate_fn=my_collate_fn,
)

# Named access
print(result.code_length)

# Access fields
model, code_length, history = result.model, result.code_length, result.history
```

### MIR encoding

```python
from preqtorch import MIREncoder

encoder = MIREncoder(model_class=model_class)

result = encoder.encode(
    dataloader=my_loader,
    set_name="My Dataset",
    n_replay_samples=2,
    learning_rate=0.001,
    seed=42,
    alpha=0.1,
    collate_fn=my_collate_fn,
    use_beta=True,
    use_ema=True,
    replay_type="buffer",
)

# Named access
print(result.beta, result.replay)

# Access fields
model, code_length, history, ema_params, beta, replay = result.model, result.code_length, result.history, result.ema_params, result.beta, result.replay
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## See also

Bornschein, J., Li, Y., & Hutter, M. (2022). Sequential learning of neural networks for prequential mdl. arXiv preprint arXiv:2210.07931.

Blier, L., & Ollivier, Y. (2018). The description length of deep learning models. Advances in Neural Information Processing Systems, 31.
