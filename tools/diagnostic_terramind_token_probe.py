"""
TerraMind / THOR token signal probe — does the coarse token at grid cell
(i, j) carry information about the height / cover of the 16x16 pixel patch it
spatially covers?

Why this exists
---------------
TerraMind S2 produces 768-d tokens at 16x16 over a 256x256 ROI (one token per
16x16 pixel patch). Direct fusion of these tokens via broadcast / FiLM / gates
into the AE+Tessera trunk has regressed against the no-fusion baseline. Before
investing in a Perceiver-style cross-attention decoder, we want to know whether
the tokens encode height-relevant content at all at the scale they live at.

Three patch-level probes (fit on TRAIN ids, eval on VAL ids):
  (S) Sanity / cover-type:  ridge token -> [veg_frac, build_frac, water_frac]
      Tokens should at least be able to recover land cover; if R^2 here is
      low, tokens are spatially misaligned with the labels and nothing else
      will work.
  (R) Height regression:    ridge token -> mean_height on veg-rich patches.
      Reports R^2 and per-bin RMSE. This is the LINEAR ceiling for
      "TerraMind token alone -> patch height".
  (A) Tall vs short patch:  logistic on veg-rich patches with
      mean_h > tall_thresh vs mean_h < short_thresh. Reports ROC-AUC.
      AUC < 0.65 means the token cannot even rank tall above short, so
      cross-attn is unlikely to pull height out of it.

Optional baselines via --source:
  terramind_s2 (default), thor_s2, alphaearth_pool16, tessera_pool16.
  The pooled variants mean-pool the dense pixel embeddings down to 16x16 so
  the comparison is at the same spatial granularity as TerraMind tokens.

Usage:
    python tools/diagnostic_terramind_token_probe.py
    python tools/diagnostic_terramind_token_probe.py --source thor_s2 --n-ids-train 150
    python tools/diagnostic_terramind_token_probe.py --source alphaearth_pool16   # baseline
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import rasterio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, REPO_DIR)

from tools.diagnostic_height_rmse import (  # noqa: E402
    CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT,
    LABEL_THRESHOLD, build_label_map, normalize_core_id,
)

DATA_DIR = "/u/dingqi2/workspace/esa/data/train"

# name -> (directory, glob pattern, is_token_grid, native_grid_size)
SOURCES = {
    "terramind_s2":     (os.path.join(DATA_DIR, "terramind_s2_emb"), "s2_*_embeddings.tif", True,  16),
    "terramind_s1":     (os.path.join(DATA_DIR, "terramind_s1_emb"), "s1_*_embeddings.tif", True,  16),
    "thor_s2":          (os.path.join(DATA_DIR, "thor_s2_emb"),      "s2_*_embedding.tif",  True,  16),
    "thor_s1":          (os.path.join(DATA_DIR, "thor_s1_emb"),      "s1_*_embedding.tif",  True,  16),
    # Pixel-level baselines, mean-pooled to the same 16x16 grid for fair comparison.
    "alphaearth_pool16": (os.path.join(DATA_DIR, "alphaearth_emb"),  "gee_emb_*.tif",       False, 16),
    "tessera_pool16":    (os.path.join(DATA_DIR, "tessera_emb"),     "tessera_emb_*.tif",   False, 16),
}

PATCH_SIZE = 256
GRID = 16  # token grid (16x16 -> per-patch is 16x16 pixels)
PIX_PER_PATCH = PATCH_SIZE // GRID  # 16


def load_split(path):
    with open(path) as f:
        d = json.load(f)
    return list(d["train"]), list(d["val"])


def build_emb_map(src_name):
    d, pat, _, _ = SOURCES[src_name]
    m = {normalize_core_id(p): p for p in glob.glob(os.path.join(d, pat))}
    if not m:
        raise SystemExit(f"no embeddings in {d}")
    return m


def load_token_grid(emb_path, src_name):
    """Return (D, GRID, GRID) float32. For pool16 sources, mean-pool 256x256 -> 16x16."""
    with rasterio.open(emb_path) as src:
        arr = src.read().astype(np.float32)  # (C, H, W)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    _, _, _, native = SOURCES[src_name]
    is_token = SOURCES[src_name][2]
    if is_token:
        # Already (D, 16, 16). If something off, crop/pad to 16.
        c, h, w = arr.shape
        if h != native or w != native:
            hh, ww = min(h, native), min(w, native)
            out = np.zeros((c, native, native), dtype=np.float32)
            out[:, :hh, :ww] = arr[:, :hh, :ww]
            arr = out
        return arr
    # pixel-level: mean-pool to 16x16
    c, h, w = arr.shape
    hh = (h // GRID) * GRID
    ww = (w // GRID) * GRID
    arr = arr[:, :hh, :ww]
    arr = arr.reshape(c, GRID, hh // GRID, GRID, ww // GRID).mean(axis=(2, 4))
    return arr  # (C, 16, 16)


def patch_label_stats(label):
    """label: (4, 256, 256). Returns dict of per-patch (16x16) stats:
       mean_h (over height-valid pixels), veg_frac, build_frac, water_frac,
       n_height_valid, n_valid, mean_h_veg (mean h restricted to veg pixels).
    """
    valid = ~np.all(label == 0, axis=0)  # (256,256)
    has_class = (label[CH_BUILDING] + label[CH_VEGETATION] + label[CH_WATER]) > 0
    height_hole = (label[CH_HEIGHT] == 0) & has_class
    h_valid = valid & ~height_hole
    veg = label[CH_VEGETATION] > LABEL_THRESHOLD
    bld = label[CH_BUILDING] > LABEL_THRESHOLD
    wtr = label[CH_WATER] > LABEL_THRESHOLD

    # Reshape into (GRID, PIX, GRID, PIX) and aggregate over the two PIX dims.
    def block(x):
        return x.reshape(GRID, PIX_PER_PATCH, GRID, PIX_PER_PATCH)

    h = label[CH_HEIGHT]
    h_valid_b  = block(h_valid).sum(axis=(1, 3)).astype(np.int32)
    valid_b    = block(valid).sum(axis=(1, 3)).astype(np.int32)
    veg_b      = block(veg).sum(axis=(1, 3)).astype(np.int32)
    bld_b      = block(bld).sum(axis=(1, 3)).astype(np.int32)
    wtr_b      = block(wtr).sum(axis=(1, 3)).astype(np.int32)

    h_masked = np.where(h_valid, h, 0.0)
    sum_h_b  = block(h_masked).sum(axis=(1, 3))
    mean_h_b = np.where(h_valid_b > 0, sum_h_b / np.maximum(h_valid_b, 1), 0.0)

    h_veg_mask = h_valid & veg
    h_veg_only = np.where(h_veg_mask, h, 0.0)
    sum_h_veg_b = block(h_veg_only).sum(axis=(1, 3))
    n_h_veg_b   = block(h_veg_mask).sum(axis=(1, 3)).astype(np.int32)
    mean_h_veg_b = np.where(n_h_veg_b > 0, sum_h_veg_b / np.maximum(n_h_veg_b, 1), 0.0)

    PX = PIX_PER_PATCH * PIX_PER_PATCH
    return {
        "veg_frac":   veg_b   / PX,
        "build_frac": bld_b   / PX,
        "water_frac": wtr_b   / PX,
        "n_height_valid": h_valid_b,
        "n_height_valid_veg": n_h_veg_b,
        "n_valid": valid_b,
        "mean_h": mean_h_b,
        "mean_h_veg": mean_h_veg_b,
    }


def apply_orientation(tok, orientation):
    """tok: (D, GRID, GRID). Returns (D, GRID, GRID) after spatial transform."""
    if orientation == "identity":
        return tok
    if orientation == "transpose":
        return tok.transpose(0, 2, 1)
    if orientation == "flip_y":
        return tok[:, ::-1, :].copy()
    if orientation == "flip_x":
        return tok[:, :, ::-1].copy()
    if orientation == "rot90":
        return np.rot90(tok, k=1, axes=(1, 2)).copy()
    if orientation == "rot180":
        return np.rot90(tok, k=2, axes=(1, 2)).copy()
    if orientation == "rot270":
        return np.rot90(tok, k=3, axes=(1, 2)).copy()
    raise ValueError(orientation)


def collect_pool(ids, label_map, emb_map, src_name, max_ids, orientation="identity", log_prefix=""):
    Xs, COVs, MHs, MHVs, NHVs = [], [], [], [], []
    n_ids_used = 0
    t0 = time.time()
    for i, cid in enumerate(ids):
        if max_ids and n_ids_used >= max_ids:
            break
        if cid not in label_map or cid not in emb_map:
            continue
        with rasterio.open(label_map[cid]) as src:
            label = src.read().astype(np.float32)
        if label.shape[1] < PATCH_SIZE or label.shape[2] < PATCH_SIZE:
            continue
        label = label[:, :PATCH_SIZE, :PATCH_SIZE]
        try:
            tok = load_token_grid(emb_map[cid], src_name)  # (D, 16, 16)
        except Exception as e:
            print(f"  {log_prefix}skip {cid}: {e}")
            continue
        D, gh, gw = tok.shape
        if (gh, gw) != (GRID, GRID):
            continue
        tok = apply_orientation(tok, orientation)
        stats = patch_label_stats(label)
        x = tok.reshape(D, GRID * GRID).T  # (256, D)
        cov = np.stack([stats["veg_frac"], stats["build_frac"], stats["water_frac"]], axis=-1).reshape(GRID * GRID, 3)
        mh = stats["mean_h"].reshape(-1)
        mhv = stats["mean_h_veg"].reshape(-1)
        nhv = stats["n_height_valid_veg"].reshape(-1)
        Xs.append(x); COVs.append(cov); MHs.append(mh); MHVs.append(mhv); NHVs.append(nhv)
        n_ids_used += 1
        if (i + 1) % 50 == 0:
            print(f"  {log_prefix}{i+1}/{len(ids)} ids scanned, used={n_ids_used}, t={time.time()-t0:.1f}s")
    if not Xs:
        raise SystemExit(f"{log_prefix}collected nothing")
    print(f"  {log_prefix}done: ids_used={n_ids_used}, patches={n_ids_used*GRID*GRID:,}, "
          f"D={Xs[0].shape[1]}, t={time.time()-t0:.1f}s")
    return {
        "X":   np.concatenate(Xs, axis=0),
        "cov": np.concatenate(COVs, axis=0),
        "mh":  np.concatenate(MHs, axis=0),
        "mhv": np.concatenate(MHVs, axis=0),
        "nhv": np.concatenate(NHVs, axis=0),
    }


def per_bin_eval(pred_h, true_h, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true_h >= lo) & (true_h < hi)
        if not m.any():
            rows.append({"lo": lo, "hi": hi, "n": 0, "rmse": float("nan"), "bias": float("nan")})
            continue
        d = (pred_h[m] - true_h[m]).astype(np.float64)
        rows.append({
            "lo": lo, "hi": hi, "n": int(d.size),
            "rmse": float(np.sqrt(np.mean(d**2))),
            "bias": float(np.mean(d)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="terramind_s2", choices=list(SOURCES.keys()))
    ap.add_argument("--n-ids-train", type=int, default=200)
    ap.add_argument("--n-ids-eval",  type=int, default=100)
    ap.add_argument("--min-veg-frac",     type=float, default=0.30,
                    help="patch is 'veg-rich' if >= this fraction of veg pixels")
    ap.add_argument("--min-height-valid", type=int, default=64,
                    help="patch must have at least this many height-valid VEG pixels for height probes")
    ap.add_argument("--tall-thresh",  type=float, default=25.0)
    ap.add_argument("--short-thresh", type=float, default=8.0)
    ap.add_argument("--ridge-alphas", type=str, default="0.1,1,10,100,1000,10000",
                    help="comma-separated alphas for RidgeCV")
    ap.add_argument("--clip-std", type=float, default=10.0,
                    help="clip standardized features to +-this (guards against tiny-std channels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-name", default=None)
    ap.add_argument("--token-orientation", default="identity",
                    choices=["identity", "transpose", "flip_y", "flip_x", "rot90", "rot180", "rot270"],
                    help="diagnostic: apply spatial transform to token grid before probing")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_name = args.output_name or f"_terramind_token_probe_{args.source}"
    out_dir = os.path.join(REPO_DIR, "runs", out_name)
    os.makedirs(out_dir, exist_ok=True)

    train_ids, val_ids = load_split(os.path.join(REPO_DIR, "splits", "split.json"))
    rng.shuffle(train_ids); rng.shuffle(val_ids)

    label_map = build_label_map(os.path.join(DATA_DIR, "labels"))
    emb_map = build_emb_map(args.source)

    print(f"source         : {args.source}")
    print(f"train ids cap  : {args.n_ids_train}  | val ids cap : {args.n_ids_eval}")
    print(f"veg-rich patch : veg_frac >= {args.min_veg_frac}")
    print(f"height-valid   : >= {args.min_height_valid} veg pixels with height in patch")
    print(f"tall > {args.tall_thresh}m  short < {args.short_thresh}m  (mean h over veg pixels in patch)")
    print()

    print(f"token orientation: {args.token_orientation}\n")
    print("[collect train pool]")
    pool_tr = collect_pool(train_ids, label_map, emb_map, args.source,
                           max_ids=args.n_ids_train,
                           orientation=args.token_orientation, log_prefix="train ")
    print("[collect val pool]")
    pool_va = collect_pool(val_ids, label_map, emb_map, args.source,
                           max_ids=args.n_ids_eval,
                           orientation=args.token_orientation, log_prefix="val   ")

    # Standardize features once.
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, roc_auc_score

    X_tr_raw = pool_tr["X"]; X_va_raw = pool_va["X"]
    # Feature diagnostics on raw TRAIN features
    raw_mean = X_tr_raw.mean(axis=0); raw_std = X_tr_raw.std(axis=0)
    print(f"\n  raw train feature stats: D={X_tr_raw.shape[1]}  "
          f"mean abs range=[{np.abs(raw_mean).min():.3e}, {np.abs(raw_mean).max():.3e}]  "
          f"std range=[{raw_std.min():.3e}, {raw_std.max():.3e}]  "
          f"n_zero_std={(raw_std == 0).sum()}  any_nonfinite={(~np.isfinite(X_tr_raw)).any()}")

    sc = StandardScaler().fit(X_tr_raw)
    X_tr = sc.transform(X_tr_raw)
    X_va = sc.transform(X_va_raw)
    # Tiny-std channels can blow up after scaling; clip to a sane range.
    np.clip(X_tr, -args.clip_std, args.clip_std, out=X_tr)
    np.clip(X_va, -args.clip_std, args.clip_std, out=X_va)
    # Replace any remaining nonfinite (from zero-std channels mapped to 0/0).
    X_tr = np.where(np.isfinite(X_tr), X_tr, 0.0)
    X_va = np.where(np.isfinite(X_va), X_va, 0.0)

    alphas = [float(a) for a in args.ridge_alphas.split(",")]
    results = {"source": args.source, "D": int(X_tr.shape[1]),
               "n_train_patches": int(X_tr.shape[0]),
               "n_val_patches":   int(X_va.shape[0]),
               "ridge_alphas": alphas}

    # ----- (S) Sanity: cover-type prediction -----
    print("\n" + "=" * 78)
    print("[S] Cover-type ridge: token -> [veg_frac, build_frac, water_frac]")
    print("=" * 78)
    cov_tr = pool_tr["cov"]; cov_va = pool_va["cov"]
    rs = RidgeCV(alphas=alphas).fit(X_tr, cov_tr)
    pred = rs.predict(X_va)
    r2s = {name: float(r2_score(cov_va[:, i], pred[:, i]))
           for i, name in enumerate(["veg_frac", "build_frac", "water_frac"])}
    print(f"  RidgeCV alpha = {rs.alpha_:.3g}")
    for k, v in r2s.items():
        print(f"  R^2 {k:12s} = {v:+.4f}")
    results["cover_R2"] = r2s
    results["cover_alpha"] = float(rs.alpha_)

    # ----- (R) Height regression on veg-rich patches -----
    print("\n" + "=" * 78)
    print(f"[R] Height ridge: token -> mean_h_veg (veg_frac>={args.min_veg_frac}, "
          f"n_height_valid_veg>={args.min_height_valid})")
    print("=" * 78)
    def veg_mask(pool):
        return (pool["cov"][:, 0] >= args.min_veg_frac) & (pool["nhv"] >= args.min_height_valid)
    m_tr = veg_mask(pool_tr); m_va = veg_mask(pool_va)
    print(f"  veg-rich patches  train={m_tr.sum():,}/{m_tr.size:,}  val={m_va.sum():,}/{m_va.size:,}")
    if m_tr.sum() < 200 or m_va.sum() < 200:
        print("  WARN: too few veg-rich patches; results will be noisy.")
    yh_tr = pool_tr["mhv"][m_tr]; yh_va = pool_va["mhv"][m_va]
    rh = RidgeCV(alphas=alphas).fit(X_tr[m_tr], yh_tr)
    yh_pred = rh.predict(X_va[m_va])
    r2_h = float(r2_score(yh_va, yh_pred))
    rmse_h = float(np.sqrt(np.mean((yh_pred - yh_va) ** 2)))
    print(f"  RidgeCV alpha = {rh.alpha_:.3g}")
    print(f"  R^2 mean_h_veg  = {r2_h:+.4f}")
    print(f"  RMSE mean_h_veg = {rmse_h:.3f} m")
    edges = [0, 5, 10, 20, 30, 50, 200]
    bins = per_bin_eval(yh_pred, yh_va, edges)
    print("  per-bin RMSE / bias (val):")
    for r in bins:
        print(f"    [{r['lo']:>3}-{r['hi']:>3}) m  n={r['n']:>6}  rmse={r['rmse']:.3f}  bias={r['bias']:+.3f}")
    results["height_R2"]    = r2_h
    results["height_RMSE"]  = rmse_h
    results["height_alpha"] = float(rh.alpha_)
    results["height_bins"]  = bins

    # ----- (A) Tall vs short veg patches -----
    print("\n" + "=" * 78)
    print(f"[A] Logistic AUC: tall (mean_h_veg>{args.tall_thresh}) vs short "
          f"(mean_h_veg<{args.short_thresh}) on veg-rich patches")
    print("=" * 78)
    def tall_short(pool, m):
        h = pool["mhv"][m]
        return (h > args.tall_thresh), (h < args.short_thresh)
    t_tr, s_tr = tall_short(pool_tr, m_tr)
    t_va, s_va = tall_short(pool_va, m_va)
    print(f"  train tall={t_tr.sum():,}  short={s_tr.sum():,}")
    print(f"  val   tall={t_va.sum():,}  short={s_va.sum():,}")
    if min(t_tr.sum(), s_tr.sum(), t_va.sum(), s_va.sum()) < 50:
        print("  WARN: very few tall or short patches; AUC will be noisy.")
        results["auc_tall_short"] = None
    else:
        Xa_tr = np.concatenate([X_tr[m_tr][t_tr], X_tr[m_tr][s_tr]])
        ya_tr = np.concatenate([np.ones(t_tr.sum()), np.zeros(s_tr.sum())])
        Xa_va = np.concatenate([X_va[m_va][t_va], X_va[m_va][s_va]])
        ya_va = np.concatenate([np.ones(t_va.sum()), np.zeros(s_va.sum())])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
        clf.fit(Xa_tr, ya_tr)
        score = clf.decision_function(Xa_va)
        auc = float(roc_auc_score(ya_va, score))
        print(f"  ROC-AUC tall_vs_short = {auc:.4f}")
        results["auc_tall_short"] = auc

    # Trivial baseline: predict mean of TRAIN heights
    yh_const = np.full_like(yh_va, fill_value=float(np.mean(yh_tr)))
    rmse_const = float(np.sqrt(np.mean((yh_const - yh_va) ** 2)))
    results["height_RMSE_constant_baseline"] = rmse_const
    print(f"\n  baseline RMSE (predict train mean) = {rmse_const:.3f} m  "
          f"=> probe lifts RMSE by {rmse_const - rmse_h:+.3f} m")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
