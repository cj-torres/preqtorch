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

    def forward(self, x):
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

def test_move_to_device_tensor():
    """Test if _move_to_device correctly moves tensors."""
    # Initialize encoder
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

    # Create a tensor on CPU
    cpu_tensor = torch.randn(5, 10)
    assert cpu_tensor.device.type == 'cpu'

    # Move encoder to CUDA
    encoder.to('cuda')

    # Move tensor to device
    cuda_tensor = encoder._move_to_device(cpu_tensor)

    # Check if tensor is on CUDA
    assert cuda_tensor.device.type == 'cuda'

def test_move_to_device_tuple():
    """Test if _move_to_device correctly moves tuples of tensors."""
    # Initialize encoder
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

    # Create tensors on CPU
    tensor1 = torch.randn(5, 10)
    tensor2 = torch.randn(5, 5)
    cpu_tuple = (tensor1, tensor2)

    # Move encoder to CUDA
    encoder.to('cuda')

    # Move tuple to device
    cuda_tuple = encoder._move_to_device(cpu_tuple)

    # Check if tensors in tuple are on CUDA
    assert cuda_tuple[0].device.type == 'cuda'
    assert cuda_tuple[1].device.type == 'cuda'

def test_move_to_device_nested_tuple():
    """Test if _move_to_device correctly moves nested tuples of tensors."""
    # Initialize encoder
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

    # Create tensors on CPU
    tensor1 = torch.randn(5, 10)
    tensor2 = torch.randn(5, 5)
    tensor3 = torch.randn(3, 3)
    nested_tuple = (tensor1, (tensor2, tensor3))

    # Move encoder to CUDA
    encoder.to('cuda')

    # Move nested tuple to device
    cuda_nested_tuple = encoder._move_to_device(nested_tuple)

    # Check if tensors in nested tuple are on CUDA
    assert cuda_nested_tuple[0].device.type == 'cuda'
    assert cuda_nested_tuple[1][0].device.type == 'cuda'
    assert cuda_nested_tuple[1][1].device.type == 'cuda'

def test_move_to_device_list():
    """Test if _move_to_device correctly moves lists of tensors."""
    # Initialize encoder
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

    # Create tensors on CPU
    tensor1 = torch.randn(5, 10)
    tensor2 = torch.randn(5, 5)
    cpu_list = [tensor1, tensor2]

    # Move encoder to CUDA
    encoder.to('cuda')

    # Move list to device
    cuda_list = encoder._move_to_device(cpu_list)

    # Check if tensors in list are on CUDA
    assert cuda_list[0].device.type == 'cuda'
    assert cuda_list[1].device.type == 'cuda'

def test_move_to_device_dict():
    """Test if _move_to_device correctly moves dictionaries of tensors."""
    # Initialize encoder
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

    # Create tensors on CPU
    tensor1 = torch.randn(5, 10)
    tensor2 = torch.randn(5, 5)
    cpu_dict = {'tensor1': tensor1, 'tensor2': tensor2}

    # Move encoder to CUDA
    encoder.to('cuda')

    # Move dictionary to device
    cuda_dict = encoder._move_to_device(cpu_dict)

    # Check if tensors in dictionary are on CUDA
    assert cuda_dict['tensor1'].device.type == 'cuda'
    assert cuda_dict['tensor2'].device.type == 'cuda'

def test_move_to_cpu_tensor():
    """Test if _move_to_cpu correctly moves tensors."""
    # Initialize encoder
    model_class = ModelClass(
        model=SimpleModel,
        device='cuda',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    encoder = PrequentialEncoder(model_class=model_class, device='cuda')

    # Create a tensor on CUDA
    cuda_tensor = torch.randn(5, 10, device='cuda')
    assert cuda_tensor.device.type == 'cuda'

    # Move tensor to CPU
    cpu_tensor = encoder._move_to_cpu(cuda_tensor)

    # Check if tensor is on CPU
    assert cpu_tensor.device.type == 'cpu'

def test_move_to_cpu_tuple():
    """Test if _move_to_cpu correctly moves tuples of tensors."""
    # Initialize encoder
    model_class = ModelClass(
        model=SimpleModel,
        device='cuda',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    encoder = PrequentialEncoder(model_class=model_class, device='cuda')

    # Create tensors on CUDA
    tensor1 = torch.randn(5, 10, device='cuda')
    tensor2 = torch.randn(5, 5, device='cuda')
    cuda_tuple = (tensor1, tensor2)

    # Move tuple to CPU
    cpu_tuple = encoder._move_to_cpu(cuda_tuple)

    # Check if tensors in tuple are on CPU
    assert cpu_tuple[0].device.type == 'cpu'
    assert cpu_tuple[1].device.type == 'cpu'

def test_move_to_cpu_nested_tuple():
    """Test if _move_to_cpu correctly moves nested tuples of tensors."""
    # Initialize encoder
    model_class = ModelClass(
        model=SimpleModel,
        device='cuda',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    encoder = PrequentialEncoder(model_class=model_class, device='cuda')

    # Create tensors on CUDA
    tensor1 = torch.randn(5, 10, device='cuda')
    tensor2 = torch.randn(5, 5, device='cuda')
    tensor3 = torch.randn(3, 3, device='cuda')
    nested_tuple = (tensor1, (tensor2, tensor3))

    # Move nested tuple to CPU
    cpu_nested_tuple = encoder._move_to_cpu(nested_tuple)

    # Check if tensors in nested tuple are on CPU
    assert cpu_nested_tuple[0].device.type == 'cpu'
    assert cpu_nested_tuple[1][0].device.type == 'cpu'
    assert cpu_nested_tuple[1][1].device.type == 'cpu'

def test_move_to_cpu_list():
    """Test if _move_to_cpu correctly moves lists of tensors."""
    # Initialize encoder
    model_class = ModelClass(
        model=SimpleModel,
        device='cuda',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    encoder = PrequentialEncoder(model_class=model_class, device='cuda')

    # Create tensors on CUDA
    tensor1 = torch.randn(5, 10, device='cuda')
    tensor2 = torch.randn(5, 5, device='cuda')
    cuda_list = [tensor1, tensor2]

    # Move list to CPU
    cpu_list = encoder._move_to_cpu(cuda_list)

    # Check if tensors in list are on CPU
    assert cpu_list[0].device.type == 'cpu'
    assert cpu_list[1].device.type == 'cpu'

def test_move_to_cpu_dict():
    """Test if _move_to_cpu correctly moves dictionaries of tensors."""
    # Initialize encoder
    model_class = ModelClass(
        model=SimpleModel,
        device='cuda',
        kwargs={
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    )

    encoder = PrequentialEncoder(model_class=model_class, device='cuda')

    # Create tensors on CUDA
    tensor1 = torch.randn(5, 10, device='cuda')
    tensor2 = torch.randn(5, 5, device='cuda')
    cuda_dict = {'tensor1': tensor1, 'tensor2': tensor2}

    # Move dictionary to CPU
    cpu_dict = encoder._move_to_cpu(cuda_dict)

    # Check if tensors in dictionary are on CPU
    assert cpu_dict['tensor1'].device.type == 'cpu'
    assert cpu_dict['tensor2'].device.type == 'cpu'

def test_move_to_cpu_when_already_on_cpu():
    """Test if _move_to_cpu correctly handles tensors already on CPU."""
    # Initialize encoder with CPU
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

    # Create a tensor on CPU
    cpu_tensor = torch.randn(5, 10)
    assert cpu_tensor.device.type == 'cpu'

    # Move tensor to CPU (should be a no-op)
    result_tensor = encoder._move_to_cpu(cpu_tensor)

    # Check if tensor is still on CPU and is the same object
    assert result_tensor.device.type == 'cpu'
    assert result_tensor is cpu_tensor  # Should be the same object, not a copy

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
