from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class PrequentialBatch:
    inputs: Any
    targets: Any
    output_mask: Any
    target_mask: Any

    def to(self, device, non_blocking=False):
        return PrequentialBatch(
            inputs=_move(self.inputs, device, non_blocking),
            targets=_move(self.targets, device, non_blocking),
            output_mask=_move(self.output_mask, device, non_blocking),
            target_mask=_move(self.target_mask, device, non_blocking),
        )


def _move(obj, device, non_blocking=False):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, tuple):
        return tuple(_move(x, device, non_blocking) for x in obj)
    if isinstance(obj, list):
        return [_move(x, device, non_blocking) for x in obj]
    if isinstance(obj, dict):
        return {k: _move(v, device, non_blocking) for k, v in obj.items()}
    return obj


def _validate_tensor_or_tuple_of_tensors(value, name):
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, tuple) and all(isinstance(x, torch.Tensor) for x in value):
        return
    raise TypeError(f"{name} must return a torch.Tensor or tuple of torch.Tensor values")


class PrequentialDataset(Dataset):
    def __init__(self, inputs, targets, masks=None, target_masks=None):
        self.inputs = inputs
        self.targets = targets
        self.masks = masks
        self.target_masks = target_masks

        for name, obj in (("inputs", inputs), ("targets", targets)):
            if not hasattr(obj, "__len__") or not hasattr(obj, "__getitem__"):
                raise TypeError(f"{name} must be indexable (__len__ and __getitem__ required)")

        n = len(inputs)
        if len(targets) != n:
            raise ValueError("inputs and targets must have the same length")

        if masks is not None:
            if not hasattr(masks, "__len__") or not hasattr(masks, "__getitem__"):
                raise TypeError("masks must be indexable")
            if len(masks) != n:
                raise ValueError("masks must have the same length as inputs")

        if target_masks is not None:
            if not hasattr(target_masks, "__len__") or not hasattr(target_masks, "__getitem__"):
                raise TypeError("target_masks must be indexable")
            if len(target_masks) != n:
                raise ValueError("target_masks must have the same length as inputs")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        _validate_tensor_or_tuple_of_tensors(x, "inputs")
        _validate_tensor_or_tuple_of_tensors(y, "targets")

        if self.masks is None and self.target_masks is None:
            return x, y

        if self.masks is not None and self.target_masks is None:
            m = self.masks[idx]
            _validate_tensor_or_tuple_of_tensors(m, "masks")
            return x, y, m

        out_m = self.masks[idx] if self.masks is not None else None
        tgt_m = self.target_masks[idx]
        if out_m is not None:
            _validate_tensor_or_tuple_of_tensors(out_m, "masks")
        _validate_tensor_or_tuple_of_tensors(tgt_m, "target_masks")
        return x, y, out_m, tgt_m


class PrequentialDataLoader(DataLoader):
    """DataLoader specializing prequential batch schemas from indexable sources."""

    def __init__(self, *, inputs, targets, masks=None, target_masks=None, batch_size=1, shuffle=False, pin_memory=False, **kwargs):
        dataset = PrequentialDataset(inputs, targets, masks=masks, target_masks=target_masks)
        collate_fn = kwargs.pop("collate_fn", None) or prequential_collate
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn, **kwargs)


def prequential_collate(batch: Iterable):
    from torch.utils.data._utils.collate import default_collate

    packed = default_collate(batch)
    if len(packed) == 2:
        inputs, targets = packed
        output_mask = torch.ones_like(inputs, dtype=torch.bool)
        target_mask = torch.ones_like(targets, dtype=torch.bool)
    elif len(packed) == 3:
        inputs, targets, shared_mask = packed
        output_mask = shared_mask
        target_mask = shared_mask
    elif len(packed) == 4:
        inputs, targets, output_mask, target_mask = packed
        if output_mask is None:
            output_mask = torch.ones_like(inputs, dtype=torch.bool)
    else:
        raise ValueError("Batch must collate to (inputs, targets[, mask|output_mask,target_mask]).")
    return PrequentialBatch(inputs=inputs, targets=targets, output_mask=output_mask, target_mask=target_mask)
