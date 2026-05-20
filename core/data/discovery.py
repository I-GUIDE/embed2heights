"""File discovery and source matching helpers."""

import glob
import json
import os
import re


_EMB_PREFIXES = ("gee_emb_", "tessera_emb_", "emb_", "s2_", "s1_")
_EMB_SUFFIXES = ("_embedding", "_embeddings", "_quantized", "_merged")
_YEAR_RE = re.compile(r"_\d{4}$")


def _strip_prefixes(base, prefixes):
    for prefix in prefixes:
        if base.startswith(prefix):
            return base[len(prefix):]
    return base


def _strip_suffixes(base, suffixes):
    for suffix in suffixes:
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


def normalize_core_id(filename):
    """
    Core ID with year suffix stripped. Used to match embeddings to labels and
    to key a split file across sensors/years.

    Handles train/test naming, e.g.
      Train: gee_emb_0000_BE.tif, tessera_emb_0000_BE.tif, s2_0000_BE_2023_embeddings.tif
      Test:  emb_3001_BE_2023_quantized.tif, s2_3001_BE_2023_embedding.tif
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    base = _strip_prefixes(base, ("label_", "pred_"))
    base = _strip_prefixes(base, _EMB_PREFIXES)
    base = _strip_suffixes(base, _EMB_SUFFIXES)
    return _YEAR_RE.sub("", base)


def submission_id(filename):
    """
    Leaderboard submission id -- same as `normalize_core_id` but keeps the
    '_YYYY' year suffix that the submission format requires.

    Example: 'emb_3001_BE_2023_quantized.tif' -> '3001_BE_2023'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    base = _strip_prefixes(base, _EMB_PREFIXES)
    return _strip_suffixes(base, _EMB_SUFFIXES)


def find_file_pairs(emb_dir, tar_dir):
    """
    Fast and robust O(N) file matching using a hash map and regex normalization.
    Searches recursively and guarantees a match regardless of prefixes/suffixes.
    """
    pairs = []

    emb_files = glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True)
    label_map = _index_label_dir(tar_dir)

    for e_path in emb_files:
        norm_id = normalize_core_id(e_path)

        if norm_id in label_map:
            pairs.append((e_path, label_map[norm_id]))

    return pairs


def find_embedding_files(emb_dir):
    """
    List all embedding .tif files without requiring labels.
    Used for competition test-set prediction where no ground truth exists.
    Returns list of embedding file paths.
    """
    return sorted(glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True))


def _index_embedding_dir(emb_dir):
    files = find_embedding_files(emb_dir)
    return {normalize_core_id(path): path for path in files}


def _index_label_dir(tar_dir):
    label_files = glob.glob(os.path.join(tar_dir, "**", "label_*.tif"), recursive=True)
    return {normalize_core_id(path): path for path in label_files}


def _match_indexed_sources(indexes):
    common_ids = sorted(set.intersection(*(set(index) for index in indexes)))
    return [tuple(index[cid] for index in indexes) for cid in common_ids]


def _match_embedding_dirs(*emb_dirs):
    return _match_indexed_sources([_index_embedding_dir(emb_dir) for emb_dir in emb_dirs])


def _match_labeled_embedding_dirs(*emb_dirs, **kwargs):
    tar_dir = kwargs["tar_dir"]
    indexes = [_index_embedding_dir(emb_dir) for emb_dir in emb_dirs]
    indexes.append(_index_label_dir(tar_dir))
    return _match_indexed_sources(indexes)


def find_multisource_file_pairs(primary_emb_dir, secondary_emb_dir, tar_dir):
    """
    Match two pixel-aligned embedding sources and labels by normalized core id.

    Returns tuples of (primary_embedding, secondary_embedding, label). The
    primary path is used for split keys and output ids, so pass AlphaEarth as
    primary when building AlphaEarth+Tessera experiments.
    """
    return _match_labeled_embedding_dirs(primary_emb_dir, secondary_emb_dir, tar_dir=tar_dir)


def find_trisource_file_pairs(primary_emb_dir, secondary_emb_dir, token_emb_dir, tar_dir):
    """
    Match two pixel-aligned sources, one token source, and labels by normalized core id.

    Returns tuples of (primary_embedding, secondary_embedding, token_embedding, label).
    """
    return _match_labeled_embedding_dirs(
        primary_emb_dir,
        secondary_emb_dir,
        token_emb_dir,
        tar_dir=tar_dir,
    )


def find_multitoken_file_pairs(primary_emb_dir, secondary_emb_dir, token_dirs, tar_dir):
    """
    Match two pixel-aligned sources, one or more token sources, and labels.

    Returns tuples of (primary_embedding, secondary_embedding, *token_embeddings, label).
    """
    return _match_labeled_embedding_dirs(
        primary_emb_dir,
        secondary_emb_dir,
        *token_dirs,
        tar_dir=tar_dir,
    )


def find_multisource_embedding_files(primary_emb_dir, secondary_emb_dir):
    """
    Match two label-free embedding dirs by normalized core id.

    Returns tuples of (primary_embedding, secondary_embedding). The primary
    path is used for leaderboard submission ids.
    """
    return _match_embedding_dirs(primary_emb_dir, secondary_emb_dir)


def find_trisource_embedding_files(primary_emb_dir, secondary_emb_dir, token_emb_dir):
    """
    Match two pixel-aligned embedding dirs and one token embedding dir for inference.

    Returns tuples of (primary_embedding, secondary_embedding, token_embedding).
    """
    return _match_embedding_dirs(primary_emb_dir, secondary_emb_dir, token_emb_dir)


def find_multitoken_embedding_files(primary_emb_dir, secondary_emb_dir, token_dirs):
    """
    Match two pixel-aligned embedding dirs and one or more token embedding dirs.

    Returns tuples of (primary_embedding, secondary_embedding, *token_embeddings).
    """
    return _match_embedding_dirs(primary_emb_dir, secondary_emb_dir, *token_dirs)


def save_split(split_path, train_pairs, val_pairs):
    """Save train/val split to a JSON file using normalized core IDs."""
    data = {
        "train": [normalize_core_id(pair[0]) for pair in train_pairs],
        "val": [normalize_core_id(pair[0]) for pair in val_pairs],
    }
    with open(split_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Split saved to {split_path} (train={len(data['train'])}, val={len(data['val'])})")


def load_split(split_path, all_pairs):
    """Load a saved split and reconstruct file pair lists."""
    with open(split_path) as f:
        data = json.load(f)

    train_ids = set(data["train"])
    val_ids = set(data["val"])

    train_pairs, val_pairs = [], []
    for pair in all_pairs:
        core_id = normalize_core_id(pair[0])
        if core_id in train_ids:
            train_pairs.append(pair)
        elif core_id in val_ids:
            val_pairs.append(pair)

    print(f"Split loaded from {split_path} (train={len(train_pairs)}, val={len(val_pairs)})")
    return train_pairs, val_pairs
