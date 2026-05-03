"""Device selection and recursive tensor transfer helpers."""

import torch


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(batch, device, non_blocking=False):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device, non_blocking=non_blocking) for item in batch)
    if isinstance(batch, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in batch]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")
