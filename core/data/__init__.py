"""Lightweight data helpers.

Dataset classes live in ``core.data.datasets`` to avoid importing raster and
torch dependencies when callers only need file IDs or split helpers.
"""

from .discovery import (
    find_embedding_files,
    find_file_pairs,
    find_multisource_embedding_files,
    find_multisource_file_pairs,
    find_quadsource_embedding_files,
    find_quadsource_file_pairs,
    find_trisource_embedding_files,
    find_trisource_file_pairs,
    load_split,
    normalize_core_id,
    save_split,
    submission_id,
)


__all__ = [
    "find_embedding_files",
    "find_file_pairs",
    "find_multisource_embedding_files",
    "find_multisource_file_pairs",
    "find_quadsource_embedding_files",
    "find_quadsource_file_pairs",
    "find_trisource_embedding_files",
    "find_trisource_file_pairs",
    "load_split",
    "normalize_core_id",
    "save_split",
    "submission_id",
]
