"""Reproducibility helpers."""

import random

import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
