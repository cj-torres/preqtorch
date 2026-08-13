import pytest
import torch

from preqtorch import BlockEncoder, MIREncoder, ModelClass
from preqtorch.encoders import PrequentialEncoder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        x, _ = batch
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_model_class_device_moving():
    model_class = ModelClass(SimpleModel, device="cpu")
    assert model_class.device == "cpu"
    model = model_class.initialize()
    assert next(model.parameters()).device.type == "cpu"

    model_class.to("cuda")
    assert model_class.device == "cuda"
    cuda_model = model_class.initialize()
    assert next(cuda_model.parameters()).device.type == "cuda"
    assert next(model.parameters()).device.type == "cpu"


def test_prequential_encoder_device_moving():
    model_class = ModelClass(SimpleModel, device="cpu")
    encoder = PrequentialEncoder(model_class=model_class, device="cpu")

    encoder.to("cuda")

    assert encoder.device == "cuda"
    assert model_class.device == "cuda"
    assert next(model_class.initialize().parameters()).device.type == "cuda"


def test_block_encoder_device_moving():
    model_class = ModelClass(SimpleModel, device="cpu")
    encoder = BlockEncoder(model_class=model_class, device="cpu")

    encoder.to("cuda")

    assert encoder.device == "cuda"
    assert model_class.device == "cuda"


def test_mir_encoder_device_moving():
    model_class = ModelClass(SimpleModel, device="cpu")
    encoder = MIREncoder(model_class=model_class, device="cpu")

    encoder.to("cuda")

    assert encoder.device == "cuda"
    assert model_class.device == "cuda"
