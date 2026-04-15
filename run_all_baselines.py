"""
Run all 6 embedding baselines (AlphaEarth, Tessera, TerraMind-S2, THOR-S2, TerraMind-S1, THOR-S1) end-to-end.

For each baseline this script:
  1. Trains a model via train.py  (checkpoints + loss curve + viz saved under runs/<exp>/)
  2. Runs inference via predict.py (.npy predictions saved under runs/<exp>/predictions/)
  3. Appends a row to runs/all_baselines_summary.csv with status + timing

Usage:
    python run_all_baselines.py                         # train + predict all 6 baselines
    python run_all_baselines.py --only alphaearth thor  # run a subset
    python run_all_baselines.py --skip-train            # predict only (reuse existing checkpoints)
    python run_all_baselines.py --skip-predict          # train only
    python run_all_baselines.py --predict-test          # predict on competition test set (no labels)
    python run_all_baselines.py --epochs 10             # override training epochs
    python run_all_baselines.py --data-dir /path/data   # override data root

Each baseline is isolated: a failure in one does not abort the others. Full
stdout/stderr of each sub-run is tee'd to runs/<exp>/run.log.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# --- Resolve script-relative paths so cwd doesn't matter ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")
PREDICT_SCRIPT = os.path.join(SCRIPT_DIR, "predict.py")
DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "runs")

# Default data roots: ../data/train and ../data/test relative to this file
DEFAULT_TRAIN_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "train"))
DEFAULT_TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "test"))


@dataclass
class Baseline:
    key: str                     # short id used on the CLI (--only)
    experiment_name: str         # subfolder under runs/
    train_emb_subdir: str        # embedding directory name under train data-dir
    test_emb_subdir: str         # embedding directory name under test data-dir
    labels_subdir: str = "labels"
    model_type: str = "auto"     # passed through to train.py / predict.py


# Six baselines — one per embedding backbone / sensor combination.
# model_type is set explicitly because train.py's dataset router
# at train.py:182 does `if MODEL_TYPE == "lightunet"` — "auto" always
# falls through to LatentTokenDataset, which is wrong for pixel-aligned embeddings.
BASELINES: List[Baseline] = [
    Baseline(key="alphaearth",   experiment_name="alphaearth_baseline",
             train_emb_subdir="alphaearth_emb",
             test_emb_subdir="alphaearth_test_emb",
             model_type="lightunet"),
    Baseline(key="tessera",      experiment_name="tessera_baseline",
             train_emb_subdir="tessera_emb",
             test_emb_subdir="tessera_test_emb",
             model_type="lightunet"),
    Baseline(key="terramind_s2", experiment_name="terramind_s2_baseline",
             train_emb_subdir="terramind_s2_emb",
             test_emb_subdir="terramind_test_s2_emb",
             model_type="decoder_residual"),
    Baseline(key="thor_s2",      experiment_name="thor_s2_baseline",
             train_emb_subdir="thor_s2_emb",
             test_emb_subdir="thor_test_s2_emb",
             model_type="decoder_residual"),
    Baseline(key="terramind_s1", experiment_name="terramind_s1_baseline",
             train_emb_subdir="terramind_s1_emb",
             test_emb_subdir="terramind_test_s1_emb",
             model_type="decoder_residual"),
    Baseline(key="thor_s1",      experiment_name="thor_s1_baseline",
             train_emb_subdir="thor_s1_emb",
             test_emb_subdir="thor_test_s1_emb",
             model_type="decoder_residual"),
]


@dataclass
class StepResult:
    name: str
    returncode: int
    duration_s: float
    log_path: str


@dataclass
class BaselineResult:
    baseline: Baseline
    steps: List[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.returncode == 0 for s in self.steps)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_TRAIN_DATA_DIR,
                   help=f"Training data root containing <embedding>_emb/ and labels/ (default: {DEFAULT_TRAIN_DATA_DIR})")
    p.add_argument("--test-data-dir", default=DEFAULT_TEST_DATA_DIR,
                   help=f"Test data root containing <embedding>_test_emb/ (default: {DEFAULT_TEST_DATA_DIR})")
    p.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                   help=f"Root directory for runs/ output (default: {DEFAULT_BASE_DIR})")
    p.add_argument("--only", nargs="+", default=None, metavar="KEY",
                   choices=[b.key for b in BASELINES],
                   help="Run only the given baseline keys (default: all 6).")
    p.add_argument("--skip-train", action="store_true", help="Skip training, run prediction only.")
    p.add_argument("--skip-predict", action="store_true", help="Skip prediction, run training only.")
    p.add_argument("--predict-test", action="store_true",
                   help="Predict on competition test set (label-free) instead of training val set.")
    p.add_argument("--epochs", type=int, default=None, help="Override --epochs passed to train.py.")
    p.add_argument("--batch-size", type=int, default=None, help="Override --batch-size passed to train.py.")
    p.add_argument("--patch-size", type=int, default=None, help="Override --patch-size for both scripts.")
    p.add_argument("--max-samples", type=int, default=None, help="Limit predict.py to N samples per baseline.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return p.parse_args()


def run_step(name: str, cmd: List[str], log_path: str, dry_run: bool) -> StepResult:
    """Run a subprocess, tee'ing output to stdout and to log_path."""
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] >>> {name}")
    print("     $ " + " ".join(cmd))
    if dry_run:
        return StepResult(name=name, returncode=0, duration_s=0.0, log_path=log_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    start = time.time()
    with open(log_path, "a", buffering=1) as log_f:
        log_f.write(f"\n=== {name} @ {datetime.now().isoformat()} ===\n")
        log_f.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=SCRIPT_DIR, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()
    duration = time.time() - start
    print(f"     <<< {name} finished rc={proc.returncode} in {duration:.1f}s")
    return StepResult(name=name, returncode=proc.returncode, duration_s=duration, log_path=log_path)


def build_train_cmd(b: Baseline, args) -> List[str]:
    exp_dir = os.path.join(args.base_dir, b.experiment_name)
    split_file = os.path.join(exp_dir, "split.json")
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--model-type", b.model_type,
        "--output-dir", args.base_dir,
        "--experiment-name", b.experiment_name,
        "--train-embeddings-dir", os.path.join(args.data_dir, b.train_emb_subdir),
        "--train-targets-dir", os.path.join(args.data_dir, b.labels_subdir),
        "--split-file", split_file,
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.patch_size is not None:
        cmd += ["--patch-size", str(args.patch_size)]
    return cmd


def build_predict_cmd(b: Baseline, args) -> List[str]:
    if args.predict_test:
        # Label-free prediction on competition test set
        pred_subdir = "test_predictions"
        cmd = [
            sys.executable, PREDICT_SCRIPT,
            "--base-dir", args.base_dir,
            "--experiment-name", b.experiment_name,
            "--model-type", b.model_type,
            "--test-embeddings-dir", os.path.join(args.test_data_dir, b.test_emb_subdir),
            "--predictions-dir", os.path.join(args.base_dir, b.experiment_name, pred_subdir),
        ]
    else:
        # Predict on training data (val split) with labels for evaluation
        cmd = [
            sys.executable, PREDICT_SCRIPT,
            "--base-dir", args.base_dir,
            "--experiment-name", b.experiment_name,
            "--model-type", b.model_type,
            "--test-embeddings-dir", os.path.join(args.data_dir, b.train_emb_subdir),
            "--test-targets-dir", os.path.join(args.data_dir, b.labels_subdir),
        ]
    if args.patch_size is not None:
        cmd += ["--patch-size", str(args.patch_size)]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    return cmd


def preflight(b: Baseline, args) -> Optional[str]:
    """Return an error string if the baseline clearly can't run, else None."""
    emb = os.path.join(args.data_dir, b.train_emb_subdir)
    lbl = os.path.join(args.data_dir, b.labels_subdir)
    if not os.path.isdir(emb):
        return f"missing embeddings dir: {emb}"
    if not os.path.isdir(lbl):
        return f"missing labels dir: {lbl}"
    return None


def write_summary(results: List[BaselineResult], base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, "all_baselines_summary.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "experiment", "key", "emb_subdir", "model_type",
                        "status", "train_rc", "train_s", "predict_rc", "predict_s",
                        "predictions_dir"])
        ts = datetime.now().isoformat(timespec="seconds")
        for r in results:
            by_name = {s.name: s for s in r.steps}
            train = by_name.get("train")
            pred = by_name.get("predict")
            w.writerow([
                ts,
                r.baseline.experiment_name,
                r.baseline.key,
                r.baseline.train_emb_subdir,
                r.baseline.model_type,
                "ok" if r.ok else "FAIL",
                train.returncode if train else "",
                f"{train.duration_s:.1f}" if train else "",
                pred.returncode if pred else "",
                f"{pred.duration_s:.1f}" if pred else "",
                os.path.join(base_dir, r.baseline.experiment_name, "predictions"),
            ])
    return path


def main():
    args = parse_args()

    selected = [b for b in BASELINES if args.only is None or b.key in args.only]
    if not selected:
        print("No baselines selected.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(selected)} baseline(s): {[b.key for b in selected]}")
    print(f"  train-data-dir = {args.data_dir}")
    print(f"  test-data-dir  = {args.test_data_dir}")
    print(f"  base-dir       = {args.base_dir}")

    results: List[BaselineResult] = []
    overall_start = time.time()

    for b in selected:
        print("\n" + "=" * 70)
        print(f"  Baseline: {b.key}  (experiment={b.experiment_name})")
        print("=" * 70)
        exp_dir = os.path.join(args.base_dir, b.experiment_name)
        log_path = os.path.join(exp_dir, "run.log")
        r = BaselineResult(baseline=b)

        err = preflight(b, args)
        if err is not None:
            print(f"  SKIP ({err})")
            r.steps.append(StepResult(name="preflight", returncode=2, duration_s=0.0, log_path=log_path))
            results.append(r)
            continue

        if not args.skip_train:
            r.steps.append(run_step("train", build_train_cmd(b, args), log_path, args.dry_run))
            if r.steps[-1].returncode != 0:
                print(f"  train failed for {b.key}; skipping predict.")
                results.append(r)
                continue

        if not args.skip_predict:
            r.steps.append(run_step("predict", build_predict_cmd(b, args), log_path, args.dry_run))

        results.append(r)

    total = time.time() - overall_start
    summary_path = write_summary(results, args.base_dir)

    print("\n" + "=" * 70)
    print(f"  Done in {total:.1f}s. Summary -> {summary_path}")
    print("=" * 70)
    for r in results:
        status = "OK   " if r.ok else "FAIL "
        parts = " | ".join(f"{s.name}:rc={s.returncode},{s.duration_s:.0f}s" for s in r.steps) or "(no steps)"
        print(f"  [{status}] {r.baseline.key:<14} -> {parts}")

    # Non-zero exit if any baseline failed, so CI / shell loops can notice.
    if any(not r.ok for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
