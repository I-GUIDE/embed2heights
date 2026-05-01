"""
Linear probe on per-pixel embeddings: does the embedding distinguish tall
vegetation from short vegetation at all?

Two probes, fit on TRAIN ids and evaluated on VAL ids:
  (A) Logistic regression : tall (veg & h > tall_thresh) vs short (veg & h < short_thresh)
      → ROC-AUC. AUC < 0.7 means the embedding has likely saturated and no
      loss / sampler / head trick can recover what isn't there.
  (B) Ridge regression    : embedding -> log(1+h) on all vegetation pixels.
      → R² and per-bin RMSE. This is the LINEAR ceiling. Compare per-bin
      RMSE to your deep model's per-bin RMSE (from
      diagnostic_height_rmse.py): if the linear probe is already in the same
      ballpark on tall bins, the deep head is not the bottleneck — the
      embedding is.

The probes are pixel-level because both alphaearth (64ch) and tessera
(128ch) embeddings are stored at the full 256x256 label resolution.

Usage (run inside the emb2heights conda env):
    python tools/diagnostic_embedding_probe.py --embedding both
    python tools/diagnostic_embedding_probe.py --embedding tessera --n-ids-train 200
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
EMB_SOURCES = {
    # name -> (directory, glob pattern)
    "alphaearth": (os.path.join(DATA_DIR, "alphaearth_emb"), "gee_emb_*.tif"),
    "tessera":    (os.path.join(DATA_DIR, "tessera_emb"),    "tessera_emb_*.tif"),
}


def load_split(path):
    with open(path) as f:
        d = json.load(f)
    return list(d["train"]), list(d["val"])


def build_emb_maps(choice):
    sources = ["alphaearth", "tessera"] if choice == "both" else [choice]
    out = []
    for s in sources:
        d, pat = EMB_SOURCES[s]
        m = {normalize_core_id(p): p for p in glob.glob(os.path.join(d, pat))}
        if not m:
            raise SystemExit(f"no embeddings in {d}")
        out.append((s, m))
    return out


def collect_pool(ids, label_map, emb_maps, tall_thresh, short_thresh,
                 max_per_id, max_total, rng, log_prefix=""):
    """Walk ids; for each, sample vegetation pixels, return:
       veg_X (N, D), veg_h (N,) and capped tall/short subsets for the AUC probe.
    """
    veg_X_list, veg_h_list = [], []
    tall_X_list, tall_h_list = [], []
    short_X_list, short_h_list = [], []
    n_total = 0
    n_used_ids = 0
    n_dropped_nan = 0
    t0 = time.time()
    n_ids = len(ids)

    for i, cid in enumerate(ids):
        if cid not in label_map:
            continue
        emb_paths = []
        ok = True
        for _, m in emb_maps:
            if cid not in m:
                ok = False; break
            emb_paths.append(m[cid])
        if not ok:
            continue

        with rasterio.open(label_map[cid]) as src:
            label = src.read().astype(np.float32)
        valid = ~np.all(label == 0, axis=0)
        any_class = (label[CH_BUILDING] + label[CH_VEGETATION] + label[CH_WATER]) > 0
        height_hole = (label[CH_HEIGHT] == 0) & any_class
        height_valid = valid & ~height_hole
        veg_mask = (label[CH_VEGETATION] > LABEL_THRESHOLD) & height_valid
        if not veg_mask.any():
            continue

        h_pix = label[CH_HEIGHT][veg_mask]

        emb_chunks = []
        H, W = label.shape[1], label.shape[2]
        for ep in emb_paths:
            with rasterio.open(ep) as src:
                e = src.read().astype(np.float32)
            hh, ww = min(e.shape[1], H), min(e.shape[2], W)
            e = e[:, :hh, :ww]
            mm = veg_mask[:hh, :ww]
            emb_chunks.append(e[:, mm])
        # Mask alignment can drop a few pixels if shapes mismatch slightly.
        n_keep = min(c.shape[1] for c in emb_chunks)
        h_pix = h_pix[:n_keep]
        x_pix = np.concatenate([c[:, :n_keep] for c in emb_chunks], axis=0).T  # (n, D)

        # Some embedding tiles have NaN/inf at no-data pixels (boundaries,
        # cloud-masked, etc.). Drop those rows before sklearn sees them.
        finite = np.isfinite(x_pix).all(axis=1) & np.isfinite(h_pix)
        if not finite.all():
            n_dropped_nan += int((~finite).sum())
            x_pix = x_pix[finite]
            h_pix = h_pix[finite]
        if x_pix.shape[0] == 0:
            continue

        if x_pix.shape[0] > max_per_id:
            idx = rng.choice(x_pix.shape[0], size=max_per_id, replace=False)
            x_pix, h_pix = x_pix[idx], h_pix[idx]

        veg_X_list.append(x_pix)
        veg_h_list.append(h_pix)
        n_total += x_pix.shape[0]
        n_used_ids += 1

        m_tall = h_pix > tall_thresh
        m_short = h_pix < short_thresh
        if m_tall.any():
            tall_X_list.append(x_pix[m_tall]); tall_h_list.append(h_pix[m_tall])
        if m_short.any():
            short_X_list.append(x_pix[m_short]); short_h_list.append(h_pix[m_short])

        if (i + 1) % 50 == 0 or (i + 1) == n_ids:
            print(f"  {log_prefix}{i+1}/{n_ids} ids, used={n_used_ids}, "
                  f"veg_pixels={n_total:,}, dropped_nan={n_dropped_nan:,}, "
                  f"t={time.time()-t0:.1f}s")

        if n_total >= max_total:
            print(f"  {log_prefix}reached max_total={max_total:,}, stopping")
            break

    def _cat(parts, dim):
        if not parts:
            return np.empty((0, dim), dtype=np.float32) if dim > 0 else np.empty(0)
        return np.concatenate(parts, axis=0)

    D = veg_X_list[0].shape[1] if veg_X_list else 0
    return {
        "veg_X":   _cat(veg_X_list, D),
        "veg_h":   _cat(veg_h_list, 0),
        "tall_X":  _cat(tall_X_list, D),
        "tall_h":  _cat(tall_h_list, 0),
        "short_X": _cat(short_X_list, D),
        "short_h": _cat(short_h_list, 0),
    }


def cap_balanced(X_a, X_b, cap, rng):
    """Cap two pools to at most `cap` rows each; preserves both pools."""
    out = []
    for X in (X_a, X_b):
        if X.shape[0] > cap:
            idx = rng.choice(X.shape[0], size=cap, replace=False)
            X = X[idx]
        out.append(X)
    return out


def per_bin_eval(pred_h, true_h, edges):
    n, rmse, bias = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true_h >= lo) & (true_h < hi)
        if not m.any():
            n.append(0); rmse.append(float("nan")); bias.append(float("nan")); continue
        d = (pred_h[m] - true_h[m]).astype(np.float64)
        n.append(int(d.size))
        rmse.append(float(np.sqrt(np.mean(d**2))))
        bias.append(float(np.mean(d)))
    return n, rmse, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", choices=["alphaearth", "tessera", "both"], default="both")
    ap.add_argument("--tall-thresh", type=float, default=30.0)
    ap.add_argument("--short-thresh", type=float, default=10.0)
    ap.add_argument("--n-ids-train", type=int, default=200,
                    help="cap on train ids visited (0 = use all)")
    ap.add_argument("--n-ids-eval", type=int, default=100,
                    help="cap on val ids visited (0 = use all)")
    ap.add_argument("--max-per-id", type=int, default=1500,
                    help="cap on vegetation pixels sampled per id")
    ap.add_argument("--max-total-train", type=int, default=300_000)
    ap.add_argument("--max-total-eval",  type=int, default=150_000)
    ap.add_argument("--auc-cap", type=int, default=80_000,
                    help="cap on tall and on short pixels for the AUC probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-name", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_name = args.output_name or f"_embedding_probe_{args.embedding}"
    out_dir = os.path.join(REPO_DIR, "runs", out_name)
    os.makedirs(out_dir, exist_ok=True)

    train_ids, val_ids = load_split(os.path.join(REPO_DIR, "splits", "split.json"))
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    if args.n_ids_train > 0:
        train_ids = train_ids[: args.n_ids_train]
    if args.n_ids_eval > 0:
        val_ids = val_ids[: args.n_ids_eval]

    label_map = build_label_map(os.path.join(DATA_DIR, "labels"))
    emb_maps = build_emb_maps(args.embedding)

    print(f"embedding(s)   : {args.embedding}  (sources: {[s for s,_ in emb_maps]})")
    print(f"train ids used : {len(train_ids)}  | val ids used : {len(val_ids)}")
    print(f"tall > {args.tall_thresh}m  short < {args.short_thresh}m  (vegetation only)")
    print()

    print("Collecting train pool ...")
    pool_tr = collect_pool(
        train_ids, label_map, emb_maps,
        tall_thresh=args.tall_thresh, short_thresh=args.short_thresh,
        max_per_id=args.max_per_id, max_total=args.max_total_train,
        rng=rng, log_prefix="train ",
    )
    print(f"  veg={pool_tr['veg_h'].size:,}  tall={pool_tr['tall_h'].size:,}  "
          f"short={pool_tr['short_h'].size:,}  D={pool_tr['veg_X'].shape[1]}")

    print("\nCollecting eval pool ...")
    pool_va = collect_pool(
        val_ids, label_map, emb_maps,
        tall_thresh=args.tall_thresh, short_thresh=args.short_thresh,
        max_per_id=args.max_per_id, max_total=args.max_total_eval,
        rng=rng, log_prefix="val   ",
    )
    print(f"  veg={pool_va['veg_h'].size:,}  tall={pool_va['tall_h'].size:,}  "
          f"short={pool_va['short_h'].size:,}")

    if pool_tr['tall_h'].size < 50 or pool_va['tall_h'].size < 50:
        print(f"\nWARN: very few tall pixels (train={pool_tr['tall_h'].size}, "
              f"val={pool_va['tall_h'].size}). AUC will be noisy. Consider "
              f"--tall-thresh 20 (or smaller) or raising --n-ids-train.")

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, r2_score

    # Probe A — tall vs short
    print("\n" + "=" * 78)
    print("[A] Logistic regression: tall vs short (vegetation pixels)")
    print("=" * 78)
    Xtr_t, Xtr_s = cap_balanced(pool_tr['tall_X'], pool_tr['short_X'], args.auc_cap, rng)
    Xva_t, Xva_s = cap_balanced(pool_va['tall_X'], pool_va['short_X'], args.auc_cap, rng)
    Xtr = np.concatenate([Xtr_t, Xtr_s], axis=0)
    ytr = np.concatenate([np.ones(Xtr_t.shape[0]), np.zeros(Xtr_s.shape[0])])
    Xva = np.concatenate([Xva_t, Xva_s], axis=0)
    yva = np.concatenate([np.ones(Xva_t.shape[0]), np.zeros(Xva_s.shape[0])])
    print(f"  fit n_train = {Xtr.shape[0]:,}  eval n_val = {Xva.shape[0]:,}  D={Xtr.shape[1]}")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)
    auc = float(roc_auc_score(yva, clf.decision_function(sc.transform(Xva))))
    print(f"  AUC (val) = {auc:.4f}")

    # Probe B — linear regression on log(1+h)
    print("\n" + "=" * 78)
    print("[B] Ridge regression: emb -> log(1+h) on vegetation")
    print("=" * 78)
    sc2 = StandardScaler().fit(pool_tr['veg_X'])
    log_tr = np.log1p(pool_tr['veg_h'])
    log_va = np.log1p(pool_va['veg_h'])
    reg = Ridge(alpha=1.0)
    reg.fit(sc2.transform(pool_tr['veg_X']), log_tr)
    pred_log = reg.predict(sc2.transform(pool_va['veg_X']))
    pred_h = np.expm1(pred_log).astype(np.float32)
    r2_log = float(r2_score(log_va, pred_log))
    r2_m = float(r2_score(pool_va['veg_h'], pred_h))
    print(f"  fit n_train = {pool_tr['veg_h'].size:,}  eval n_val = {pool_va['veg_h'].size:,}")
    print(f"  R² (log space) = {r2_log:.4f}    R² (meters) = {r2_m:.4f}")

    edges = np.concatenate([np.arange(0, 40, 2.0), np.array([40.0, 50.0, 60.0, 80.0])])
    ns, rmses, biases = per_bin_eval(pred_h, pool_va['veg_h'], edges)
    print("\n  Per-bin linear-probe error (compare to your deep model's per-bin):")
    print(f"    {'bin (m)':<13} {'n':>9} {'rmse':>7} {'bias':>7}")
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if ns[i] == 0:
            continue
        print(f"    [{lo:>4.1f},{hi:>5.1f})   {ns[i]:>9,d}  {rmses[i]:>7.3f}  {biases[i]:>7.3f}")

    # Save
    out = {
        "embedding": args.embedding,
        "thresholds": {"tall": args.tall_thresh, "short": args.short_thresh},
        "n_train_ids_used": len(train_ids),
        "n_val_ids_used": len(val_ids),
        "n_train_veg_pixels": int(pool_tr['veg_h'].size),
        "n_val_veg_pixels": int(pool_va['veg_h'].size),
        "feature_dim": int(pool_tr['veg_X'].shape[1]) if pool_tr['veg_X'].size else 0,
        "probe_A_logistic": {
            "n_train_tall": int(Xtr_t.shape[0]),
            "n_train_short": int(Xtr_s.shape[0]),
            "n_val_tall": int(Xva_t.shape[0]),
            "n_val_short": int(Xva_s.shape[0]),
            "auc_val": auc,
        },
        "probe_B_ridge_logh": {
            "r2_log": r2_log, "r2_meters": r2_m,
            "per_bin": {
                "edges": edges.tolist(), "n": ns, "rmse": rmses, "bias": biases,
            },
        },
    }
    out_json = os.path.join(out_dir, "embedding_probe.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_json}")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(f"  AUC = {auc:.3f}")
    if auc >= 0.85:
        print("  → embedding HAS the categorical signal. Reweighting / balanced sampler /")
        print("    LDS / bigger head are all worth trying — bias-headroom is real.")
    elif auc >= 0.70:
        print("  → embedding has SOME signal but it's marginal. Sampler/LDS gains will be")
        print("    modest. Consider concatenating an additional embedding source (e.g.")
        print("    add tessera if you only ran alphaearth, or vice versa).")
    else:
        print("  → embedding has SATURATED on the tall tail. No loss/sampler/head trick")
        print("    will materially help. Need a different input — multi-temporal, raw")
        print("    S1/S2, or a model with higher-resolution patch tokens.")
    print(f"\n  Linear-probe R² (meters) = {r2_m:.3f}")
    print("  Compare its per-bin RMSE above to your deep model's per-bin RMSE")
    print("  (from tools/diagnostic_height_rmse.py). If linear ≈ deep on tall bins,")
    print("  head capacity is NOT the bottleneck — the embedding is.")


if __name__ == "__main__":
    main()
