"""Pre-concatenate TM-S1+TM-S2+THOR-S1+THOR-S2 token tifs into a single
3072-ch tif per tile so we can feed all 4 unused modalities as one
token source to xfusion_crosslevel (R1b) or to a new token-residual
ae_tessera_gated variant (R1c).

Each input is 768ch @ 16x16; output is 3072ch @ 16x16.

Usage:
    python tools/concat_token_modalities.py --split train
    python tools/concat_token_modalities.py --split test
"""

import argparse
import os
import re
import sys
import numpy as np
import rasterio
from rasterio.transform import Affine


SOURCES_TRAIN = {
    "tm_s1": "/projects/bcrm/emb2height/data/train/terramind_s1_emb",
    "tm_s2": "/projects/bcrm/emb2height/data/train/terramind_s2_emb",
    "thor_s1": "/projects/bcrm/emb2height/data/train/thor_s1_emb",
    "thor_s2": "/projects/bcrm/emb2height/data/train/thor_s2_emb",
}
SOURCES_TEST = {
    "tm_s1": "/projects/bcrm/emb2height/data/test/terramind_test_s1_emb",
    "tm_s2": "/projects/bcrm/emb2height/data/test/terramind_test_s2_emb",
    "thor_s1": "/projects/bcrm/emb2height/data/test/thor_test_s1_emb",
    "thor_s2": "/projects/bcrm/emb2height/data/test/thor_test_s2_emb",
}

OUT_TRAIN = "/projects/bcrm/emb2height/data/train/combined_tokens_emb"
OUT_TEST = "/projects/bcrm/emb2height/data/test/combined_tokens_test_emb"

# Tile-key pattern: e.g. s1_0000_BE_2023_embeddings.tif → key = '0000_BE'
KEY_RE = re.compile(r"^s[12]_(\d+_[A-Z]+)_\d+_embeddings?\.tif$")


def tile_key(fname):
    m = KEY_RE.match(fname)
    return m.group(1) if m else None


def index_dir(path):
    return {tile_key(f): os.path.join(path, f) for f in os.listdir(path) if tile_key(f) is not None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--max", type=int, default=None, help="Limit number of tiles (debug)")
    args = p.parse_args()

    sources = SOURCES_TRAIN if args.split == "train" else SOURCES_TEST
    out_dir = OUT_TRAIN if args.split == "train" else OUT_TEST
    os.makedirs(out_dir, exist_ok=True)

    indexed = {name: index_dir(path) for name, path in sources.items()}
    all_keys = set.intersection(*(set(d.keys()) for d in indexed.values()))
    print(f"Found {len(all_keys)} tiles common to all 4 sources in {args.split} split")
    if args.max is not None:
        all_keys = sorted(all_keys)[: args.max]

    skipped = 0
    written = 0
    for i, k in enumerate(sorted(all_keys), start=1):
        out_path = os.path.join(out_dir, f"tokens_{k}.tif")
        if os.path.exists(out_path):
            skipped += 1
            if i % 200 == 0:
                print(f"  [{i}/{len(all_keys)}] skipping existing {os.path.basename(out_path)}")
            continue

        arrs = []
        first_profile = None
        for name in ["tm_s1", "tm_s2", "thor_s1", "thor_s2"]:
            with rasterio.open(indexed[name][k]) as r:
                arr = r.read()  # (C, H, W)
                if first_profile is None:
                    first_profile = r.profile
                if arr.shape[-2:] != (16, 16):
                    print(f"  WARN: {name}/{k} unexpected spatial {arr.shape}, skipping tile", file=sys.stderr)
                    arr = None
                    break
                arrs.append(arr.astype(np.float32))
        if not arrs or len(arrs) != 4:
            continue

        concat = np.concatenate(arrs, axis=0)  # (4*768, 16, 16) = (3072, 16, 16)
        profile = dict(first_profile)
        profile.update(count=concat.shape[0], dtype="float32")

        with rasterio.open(out_path, "w", **profile) as w:
            w.write(concat)
        written += 1
        if i % 200 == 0:
            print(f"  [{i}/{len(all_keys)}] wrote {os.path.basename(out_path)} shape={concat.shape}")

    print(f"\nDone. wrote={written} skipped={skipped} out_dir={out_dir}")


if __name__ == "__main__":
    main()
