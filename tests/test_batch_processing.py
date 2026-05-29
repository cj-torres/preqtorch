import torch

from preqtorch import move_to_device


def test_move_to_device_recurses_through_arbitrary_batch_structures():
    batch = {
        "inputs": torch.tensor([1.0]),
        "metadata": [torch.tensor([2.0]), {"target": torch.tensor([3.0])}],
        "untouched": "value",
    }

    moved = move_to_device(batch, "cpu")

    assert moved["inputs"].device.type == "cpu"
    assert moved["metadata"][0].device.type == "cpu"
    assert moved["metadata"][1]["target"].device.type == "cpu"
    assert moved["untouched"] == "value"
