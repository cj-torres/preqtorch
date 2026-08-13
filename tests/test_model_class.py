import torch
import torch.nn as nn
import pytest
from preqtorch import ModelClass

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        x, _ = batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModelWithOneDimensionalParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(4))
        self.projection = nn.Linear(4, 2)

    def forward(self, batch):
        return self.projection(batch * self.scale)

def test_model_class_default_initialization():
    """Test if ModelClass properly initializes models with default initialization."""
    # Initialize ModelClass with default initialization
    model_class = ModelClass(
        model=SimpleModel,
        device='cpu',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    # Create a model instance
    model = model_class.initialize()

    # Check if model is properly initialized
    assert isinstance(model, SimpleModel)
    
    # Check if parameters are initialized (not all zeros)
    for name, param in model.named_parameters():
        if name.rsplit('.', 1)[-1] == 'bias':
            assert torch.all(param == 0).item()
        else:
            assert not torch.all(param == 0).item()

        if param.dim() > 1:
            # Xavier uniform keeps values in a reasonable range
            assert torch.max(torch.abs(param)).item() < 1.0


def test_model_class_distinguishes_biases_from_other_one_dimensional_parameters():
    model = ModelClass(ModelWithOneDimensionalParameter, device='cpu').initialize()

    assert torch.all(model.projection.bias == 0).item()
    assert not torch.all(model.scale == 0).item()

def test_model_class_custom_initialization():
    """Test if ModelClass properly uses custom initialization function."""
    
    # Define a custom initialization function
    def custom_init(model):
        # Initialize all parameters to ones
        for param in model.parameters():
            nn.init.ones_(param)
        return model
    
    # Initialize ModelClass with custom initialization
    model_class = ModelClass(
        model=SimpleModel,
        device='cpu',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        },
        init_func=custom_init
    )

    # Create a model instance
    model = model_class.initialize()

    # Check if model is properly initialized
    assert isinstance(model, SimpleModel)
    
    # Check if parameters are initialized to ones as per custom initialization
    for param in model.parameters():
        assert torch.all(param == 1.0).item()

def test_model_class_different_initializations():
    """Test if different ModelClass instances can have different initialization functions."""
    
    # Define a custom initialization function
    def custom_init(model):
        # Initialize all parameters to 2.0
        for param in model.parameters():
            nn.init.constant_(param, 2.0)
        return model
    
    # Initialize ModelClass with default initialization
    default_model_class = ModelClass(
        model=SimpleModel,
        device='cpu',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )
    
    # Initialize ModelClass with custom initialization
    custom_model_class = ModelClass(
        model=SimpleModel,
        device='cpu',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        },
        init_func=custom_init
    )

    # Create model instances
    default_model = default_model_class.initialize()
    custom_model = custom_model_class.initialize()

    # Check if models are properly initialized
    assert isinstance(default_model, SimpleModel)
    assert isinstance(custom_model, SimpleModel)
    
    # Check if default model has mixed initialization (xavier, zeros, uniform)
    has_non_two_values = False
    for param in default_model.parameters():
        if not torch.all(param == 2.0).item():
            has_non_two_values = True
            break
    assert has_non_two_values
    
    # Check if custom model has all parameters set to 2.0
    for param in custom_model.parameters():
        assert torch.all(param == 2.0).item()
