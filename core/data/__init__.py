"""Lightweight data helpers.

Dataset classes live in ``core.data.datasets`` to avoid importing raster and
torch dependencies when callers only need file IDs or split helpers.
"""

from .discovery import (
    find_embedding_files,
    find_source_pairs,
    find_source_tuples,
    load_split,
    normalize_core_id,
    save_split,
    submission_id,
)


__all__ = [
    "find_embedding_files",
    "find_source_pairs",
    "find_source_tuples",
    "load_split",
    "normalize_core_id",
    "save_split",
    "submission_id",
]
