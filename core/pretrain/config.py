"""Pretraining config persistence."""

import json
import os


def save_pretrain_config(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
