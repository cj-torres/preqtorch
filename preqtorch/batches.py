import torch


def move_to_device(obj, device, non_blocking=False):
    """Recursively move tensors in an arbitrary batch object to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device, non_blocking) for x in obj)
    if isinstance(obj, list):
        return [move_to_device(x, device, non_blocking) for x in obj]
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    return obj
