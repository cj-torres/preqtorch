import torch
import torch.nn as nn
import pytest
from preqtorch import BlockEncoder, MIREncoder, ModelClass
from preqtorch.encoders import PrequentialEncoder

# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, target_mask=None, target=None):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_model_class_device_moving():
    """Test if ModelClass properly handles device moving."""
    # Initialize ModelClass with CPU
    model_class = ModelClass(
        model=SimpleModel,
        device='cpu',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    # Check initial device
    assert model_class.device == 'cpu'

    # Create a model instance
    model = model_class.initialize()

    # Check if model is on CPU
    assert next(model.parameters()).device.type == 'cpu'

    # Move ModelClass to CUDA
    model_class.to('cuda')

    # Check if ModelClass device is updated
    assert model_class.device == 'cuda'

    # Create a new model instance
    cuda_model = model_class.initialize()

    # Check if new model is on CUDA
    assert next(cuda_model.parameters()).device.type == 'cuda'

    # The original model should still be on CPU
    assert next(model.parameters()).device.type == 'cpu'

def test_prequential_encoder_device_moving():
    """Test if PrequentialEncoder properly handles device moving."""
    # Initialize ModelClass and PrequentialEncoder with CPU
    model_class = ModelClass(
        model=SimpleModel,
        device='cpu',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    encoder = PrequentialEncoder(model_class=model_class, device='cpu')

    # Check initial device
    assert encoder.device == 'cpu'
    assert model_class.device == 'cpu'

    # Move encoder to CUDA
    encoder.to('cuda')

    # Check if both encoder and model_class devices are updated
    assert encoder.device == 'cuda'
    assert model_class.device == 'cuda'

    # Create a model and check its device
    model = model_class.initialize()
    assert next(model.parameters()).device.type == 'cuda'

def test_block_encoder_device_moving():
    """Test if BlockEncoder properly handles device moving."""
    # Initialize ModelClass and BlockEncoder with CPU
    model_class = ModelClass(
        model=SimpleModel,
        device='cpu',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    encoder = BlockEncoder(model_class=model_class, device='cpu')

    # Check initial device
    assert encoder.device == 'cpu'
    assert model_class.device == 'cpu'

    # Move encoder to CUDA
    encoder.to('cuda')

    # Check if both encoder and model_class devices are updated
    assert encoder.device == 'cuda'
    assert model_class.device == 'cuda'

def test_mir_encoder_device_moving():
    """Test if MIREncoder properly handles device moving."""
    # Initialize ModelClass and MIREncoder with CPU
    model_class = ModelClass(
        model=SimpleModel,
        device='cpu',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    encoder = MIREncoder(model_class=model_class, device='cpu')

    # Check initial device
    assert encoder.device == 'cpu'
    assert model_class.device == 'cpu'

    # Move encoder to CUDA
    encoder.to('cuda')

    # Check if both encoder and model_class devices are updated
    assert encoder.device == 'cuda'
    assert model_class.device == 'cuda'
