# PreqTorch

A PyTorch-based library for prequential coding and clustering.

## Overview

PreqTorch provides tools for prequential encoding and clustering in PyTorch. Prequential encoding is a technique for evaluating predictive models in an online learning setting, where the model is updated after each prediction.

The library includes:
- Prequential encoders (BlockEncoder, MIRSEncoder)
- Clustering algorithms based on prequential coding
- Tools for change point detection

## Installation

### From PyPI

```bash
pip install preqtorch
```

### From Source

```bash
git clone https://github.com/torrescj/preqtorch.git
cd preqtorch
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
from preqtorch import MIRSEncoder

# Define a model class
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)

# Create an encoder
encoder = MIRSEncoder(
    model_class=MyModel,
    loss_fn=torch.nn.functional.cross_entropy
)

# Encode a dataset
model, code_length, ema_params, beta, replay_streams = encoder.encode(
    dataset=my_dataset,
    set_name="My Dataset",
    n_replay_streams=2,
    learning_rate=0.001,
    batch_size=32,
    seed=42,
    alpha=0.1
)
```

### Prequential Clustering

```python
from preqtorch import MIRSEncoder, prequential_clustering

# Create an encoder
encoder = MIRSEncoder(
    model_class=MyModel,
    loss_fn=torch.nn.functional.cross_entropy
)

# Perform prequential clustering
best_seq, best_code_lengths, clusters = prequential_clustering(
    encoder=encoder,
    dataset=my_dataset,
    beam_width=3,
    learning_rate=0.001,
    seed=42,
    alpha=0.1,
    batch_size=32,
    n_replay_streams=2
)
```

## Examples

See the `tests` directory for examples of how to use the library.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.