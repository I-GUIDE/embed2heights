"""Match embedding files to labels (and to each other) across source dirs.

The data layout is fixed. Every tile has an invariant core id ``<NNNN>_<AA>``
(4 digits + 2-letter region code, e.g. ``0041_FQ``). Each source names its files
differently and only some carry a ``_YYYY`` year, e.g.

    alphaearth  gee_emb_0000_BE.tif             tessera   tessera_emb_0000_BE.tif
    terramind   s1_0000_BE_2023_embeddings.tif  thor      s1_0000_BE_2023_embedding.tif
    labels      label_0000_BE_2023.tif          test AE   emb_3001_BE_2023_quantized.tif

so everything is keyed by the year-stripped core id (that is why the no-year
AlphaEarth/Tessera tiles still line up with the year-bearing token/label tiles).
"""

import glob
import json
import os
import re

_CORE_RE = re.compile(r"\d{4}_[A-Z]{2}")
_SUBMISSION_RE = re.compile(r"\d{4}_[A-Z]{2}(?:_\d{4})?")


def normalize_core_id(filename):
    """Core id, e.g. 'label_0041_FQ_2023.tif' -> '0041_FQ'. Used to match sources
    to each other and to labels, and to key split files."""
    base = os.path.basename(filename)
    m = _CORE_RE.search(base)
    return m.group() if m else os.path.splitext(base)[0]


def submission_id(filename):
    """Leaderboard id: core id plus the '_YYYY' year when present, e.g.
    'emb_3001_BE_2023_quantized.tif' -> '3001_BE_2023'."""
    base = os.path.basename(filename)
    m = _SUBMISSION_RE.search(base)
    return m.group() if m else os.path.splitext(base)[0]


def find_embedding_files(emb_dir):
    """All embedding .tif files under a dir (label-free; for test prediction)."""
    return sorted(glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True))


def _index_embedding_dir(emb_dir):
    return {normalize_core_id(p): p for p in find_embedding_files(emb_dir)}


def _index_label_dir(tar_dir):
    label_files = glob.glob(os.path.join(tar_dir, "**", "label_*.tif"), recursive=True)
    return {normalize_core_id(p): p for p in label_files}


def _match(indexes):
    """Tuples of paths (one per index) for the core ids present in every index."""
    common = sorted(set.intersection(*(set(i) for i in indexes)))
    return [tuple(i[cid] for i in indexes) for cid in common]


def find_source_pairs(primary_emb_dir, secondary_emb_dir, token_dirs, tar_dir):
    """Match the two pixel sources + all token sources + labels by core id.

    Returns tuples (primary, secondary, *tokens, label). The primary path keys
    the split file and output ids, so pass AlphaEarth as primary.
    """
    dirs = [primary_emb_dir, secondary_emb_dir, *token_dirs]
    return _match([_index_embedding_dir(d) for d in dirs] + [_index_label_dir(tar_dir)])


def find_source_tuples(primary_emb_dir, secondary_emb_dir, token_dirs):
    """Label-free version of :func:`find_source_pairs` (for test prediction).

    Returns tuples (primary, secondary, *tokens).
    """
    dirs = [primary_emb_dir, secondary_emb_dir, *token_dirs]
    return _match([_index_embedding_dir(d) for d in dirs])


def save_split(split_path, train_pairs, val_pairs):
    """Save a train/val split to JSON using normalized core ids."""
    data = {
        "train": [normalize_core_id(p[0]) for p in train_pairs],
        "val": [normalize_core_id(p[0]) for p in val_pairs],
    }
    with open(split_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Split saved to {split_path} (train={len(data['train'])}, val={len(data['val'])})")


def load_split(split_path, all_pairs):
    """Reconstruct train/val pair lists from a saved split file."""
    with open(split_path) as f:
        data = json.load(f)
    train_ids, val_ids = set(data["train"]), set(data["val"])
    train_pairs, val_pairs = [], []
    for pair in all_pairs:
        cid = normalize_core_id(pair[0])
        if cid in train_ids:
            train_pairs.append(pair)
        elif cid in val_ids:
            val_pairs.append(pair)
    print(f"Split loaded from {split_path} (train={len(train_pairs)}, val={len(val_pairs)})")
    return train_pairs, val_pairs
