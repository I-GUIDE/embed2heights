"""Checkpoint save/load helpers."""

import torch


def state_dict_for_save(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()
