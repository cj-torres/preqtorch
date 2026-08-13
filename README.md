# PreqTorch

PreqTorch is a PyTorch package for running prequential code-length experiments with user-defined data formats, models, and loss functions.

## Installation

```bash
pip install preqtorch
```

For local development:

```bash
git clone https://github.com/cj-torres/preqtorch.git
cd preqtorch
python -m pip install -e ".[dev]"
python -m pytest
```

## Requirements

- Python 3.11+
- PyTorch 2.0+ (required for `torch.func`)

## Public API

```python
from preqtorch import (
    BlockEncoder,
    EncoderState,
    MIREncoder,
    ModelClass,
    PrequentialEncoder,
    EncoderResult,
    Replay,
    ReplayBuffer,
    ReplayStreams,
    ReplayingDataLoader,
    move_to_device,
)
```

## Data, model, and loss contract

PreqTorch treats each dataloader batch as opaque user data. A batch can be a tensor, tuple, list, dict, dataclass, or any object your model and loss function understand.

PreqTorch preserves the source dataloader's collation behavior and does not provide a default loss function. Construct PyTorch dataloaders with the batching behavior you need, and pass a `loss_fn` to encoder calls.

Models must accept the whole batch:

```python
output = model(batch)
```

Loss functions must accept the whole batch and the model output:

```python
code_lengths = loss_fn(batch, output)
```

`loss_fn` should return a tensor of code lengths or losses that can be summed. The encoders call `code_lengths.sum()` for optimization and accumulation.

## ModelClass

Encoders instantiate models through `ModelClass`. Pass a `torch.nn.Module` subclass, not an instance.
By default, matrix parameters use Xavier initialization, parameters named `bias` are zeroed, and other parameters use a uniform distribution over `[-1, 1]`.

```python
import torch
from preqtorch import ModelClass

class Classifier(torch.nn.Module):
    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, batch):
        inputs, targets = batch
        return self.linear(inputs)

model_class = ModelClass(Classifier, device="cpu", kwargs={"input_size": 10, "output_size": 2})
```

You can pass `init_func` to control initialization:

```python
def init_model(model):
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    return model

model_class = ModelClass(Classifier, device="cpu", init_func=init_model)
```

## Loss function example

```python
import torch.nn.functional as F

def loss_fn(batch, output):
    inputs, targets = batch
    return F.cross_entropy(output, targets, reduction="none")
```

For structured batches, unpack the format you defined:

```python
def sequence_loss_fn(batch, output):
    targets = batch["targets"]
    output_mask = batch["output_mask"]
    return F.cross_entropy(output[output_mask], targets[output_mask], reduction="none")
```

## Device handling

By default, encoders recursively move tensors inside standard Python containers (`tuple`, `list`, and `dict`) to the encoder device before calling `model(batch)` and `loss_fn(batch, output)`. Non-tensor objects are left unchanged.

Pass `use_device_handling=False` to `encode(...)` or `calculate_code_length(...)` if your own dataloader, model, or batch type handles device placement.

## BlockEncoder

`BlockEncoder.encode(...)` evaluates each block and accumulates its code length. After every non-final evaluation block, it restores the model's initial weights, creates a fresh optimizer, and trains on the training dataloader at the same index. That trained model is then used for the next evaluation block.

```python
from preqtorch import BlockEncoder

encoder = BlockEncoder(model_class=model_class, device="cpu")

result = encoder.encode(
    train_dataloader=[train_loader_1, train_loader_2],
    eval_dataloaders=[eval_loader_1, eval_loader_2],
    set_name="example",
    seed=42,
    loss_fn=loss_fn,
    learning_rate=1e-4,
    epochs=50,
    patience=20,
)

print(result.code_length)
print(result.history)
```

`train_dataloader` and `eval_dataloaders` must have the same length. Because training happens between evaluation blocks, the final training dataloader is not consumed.

## MIREncoder

`MIREncoder.encode(...)` processes one batched, map-style PyTorch dataloader with replay. Iterable-style datasets are not supported because replay stores and rematerializes sample indices. The encoder preserves the source dataloader's batch sampler, including sampler order and `drop_last` behavior.

```python
from preqtorch import MIREncoder

encoder = MIREncoder(model_class=model_class, device="cpu")

result = encoder.encode(
    dataloader=loader,
    set_name="example",
    n_replay_samples=2,
    loss_fn=loss_fn,
    learning_rate=1e-4,
    seed=42,
    alpha=0.1,
    use_beta=False,
    use_ema=True,
    replay_type="buffer",  # "buffer" or "streams"
)

print(result.code_length)
print(result.history)
print(result.replay)
```

Useful options:

- `n_replay_samples`: required unless `replay` is supplied. For `replay_type="buffer"`, this is the number of replay batches sampled after each source batch. For `replay_type="streams"`, it is the number of replay streams.
- `collate_fn`: overrides collation for both source and replay batches. If omitted, the source dataloader's `collate_fn` is reused.
- `replay`: a preconstructed custom `Replay` instance. It must use the exact same dataset object as the source dataloader. When supplied, it owns replay updates and sampling, and `n_replay_samples` and `replay_type` are not used.
- `shuffle`: retained for direct `initialize(...)` calls; `encode(...)` follows the source dataloader's sampler order.
- `pin_memory`: overrides pinned-memory behavior; if omitted, the encoder or source dataloader setting is reused.
- `use_device_handling=False`: disables automatic recursive tensor movement to the encoder device.
- `use_beta=True`: scales tensor outputs by a learned positive scalar. Keep this disabled for non-tensor model outputs.
- `use_ema=True`: evaluates each new source batch with an exponential moving average of the model parameters.

The built-in `ReplayBuffer` requires a fixed batch size. `ReplayStreams`, or a custom `Replay`, can be used with a variable-size custom batch sampler.

## Results

Encoder runs return an `EncoderResult` dataclass:

```python
@dataclass
class EncoderResult:
    model: Any
    code_length: float
    history: list[float]
    ema_params: Any | None = None
    beta: Any | None = None
    replay: Any | None = None
```

`BlockEncoder` fills `model`, `code_length`, and `history`.

`MIREncoder` can also fill `ema_params`, `beta`, and `replay`, depending on the options used.

## Replay utilities

PreqTorch also exports replay helpers:

- `Replay`: base class for custom replay policies. Implement `update(new_indices)` and `sample()`; `sample()` returns an iterable of `(indices, batch)` pairs.
- `ReplayBuffer`: uniformly samples up to `batch_size` items from previously seen indices.
- `ReplayStreams`: samples one batch from each replay stream, following previously seen source batches; it supports variable batch sizes.
- `ReplayingDataLoader`: wraps a map-style dataset and replay object so the current stream and replay samples can be used together; it can be iterated repeatedly and its length is the number of source batches. Direct construction requires a positive integer `batch_size`; use `ReplayingDataLoader.from_dataloader(...)` to preserve an existing dataloader's batch sampler and worker settings.

Replay implementations use PyTorch's default collation when `collate_fn` is omitted.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
