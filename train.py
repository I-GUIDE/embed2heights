"""
Train a single emb2heights backbone.

The script is deliberately monolithic: one process trains one model on one
embedding source and writes all artifacts under `runs/<experiment_name>/`.
Compose multiple training runs in a shell script or slurm array — there is no
multi-baseline driver.
"""
import os
import json
import random
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import rasterio
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.model import build_model
from core.dataset import (
    find_file_pairs, find_multisource_file_pairs, save_split, load_split,
    pick_dataset_class, MultiPixelEmbeddingDataset, AUGMENT_MODES,
)
from core.losses import ImprovedCompositeLoss, UncertaintyWeightedLoss


class ModelEMA:
    """Step-wise exponential moving average of model parameters.

    Tracked on the same device as the model. `store()`/`restore()` let the
    caller temporarily swap EMA weights in (e.g. for validation) without
    losing the live training weights.

    Decay schedule (Karras-style warmup):
        d_t = min(target_decay, (1 + t) / (warmup + t))
    The shadow tracks the live model exactly at t=0 and ramps to
    `target_decay` over ~10x `warmup` steps. Without this, `target_decay
    = 0.9995` keeps the random-init contribution above 50% for the first
    ~1400 steps, so EMA-validated runs report garbage "best" checkpoints
    until the shadow catches up. The first version of this class shipped
    without warmup and produced exactly that pathology on run 80335_1.
    """
    def __init__(self, model, decay=0.9995, warmup=10):
        self.target_decay = float(decay)
        self.warmup = float(warmup)
        self.step = 0
        # Clone current params/buffers into a detached shadow state dict.
        src = _unwrapped_state_dict(model)
        self.shadow = {k: v.detach().clone() for k, v in src.items()}
        self._backup = None

    def _current_decay(self):
        # Karras-style: at step 0, decay=0 (shadow == live); ramps to
        # target_decay as 1/(1+1/t)-style asymptote.
        warmed = (1.0 + self.step) / (self.warmup + self.step)
        return min(self.target_decay, warmed)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = self._current_decay()
        src = _unwrapped_state_dict(model)
        for k, v in src.items():
            sv = self.shadow[k]
            if v.dtype.is_floating_point:
                sv.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                # Integer buffers (e.g. num_batches_tracked) just copy through.
                sv.copy_(v.detach())

    def store(self, model):
        """Save live weights, then swap EMA weights into the model."""
        src = _unwrapped_state_dict(model)
        self._backup = {k: v.detach().clone() for k, v in src.items()}
        _load_state_dict_any(model, self.shadow)

    def restore(self, model):
        assert self._backup is not None, "restore() called without a matching store()"
        _load_state_dict_any(model, self._backup)
        self._backup = None

    def state_dict(self):
        return self.shadow


def _load_state_dict_any(model, sd):
    """Load an unwrapped state_dict into a (possibly torch.compile-wrapped) model."""
    inner = getattr(model, "_orig_mod", model)
    inner.load_state_dict(sd)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_EMB = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "train", "alphaearth_emb"))
DEFAULT_TRAIN_TAR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "train", "labels"))

MODEL_CHOICES = [
    "auto", "lightunet", "decoder_residual", "token_neck", "embedding_refiner",
    "hrnet_w18", "hrnet_w32", "tessera_iou_fusion",
]
LOSS_PRESET_CHOICES = ["auto", "current", "no_ssim_grad", "presence_centered"]

# Defaults — every one is overridable from the CLI.
DEFAULTS = {
    "experiment_name": "run01",
    "output_dir":      os.path.join(SCRIPT_DIR, "runs"),
    "batch_size":      32,
    "patch_size":      256,
    "epochs":          30,
    "lr":              2e-4,
    "weight_decay":    1e-4,
    "val_split":       0.2,
    "lambdas":         [1.0, 0.5, 0.5, 2.0],   # [MAE, SSIM, Gradient, Tversky]
    "loss_preset":     "auto",
    "presence_tversky_weight": 1.0,
    "fraction_mae_weight": 0.1,
    # Presence head is now the submission output for land-cover channels,
    # so its BCE supervision is primary, not auxiliary. Bumped from 0.25.
    "aux_weight":      1.0,
    "seed":            42,
    "model_type":      "auto",
    "grad_accum":      1,
    "num_workers":     8,
    "prefetch_factor": 4,
    "compile":         True,
}


RAW_COMPONENTS = (
    "mae",
    "fraction_mae",
    "ssim",
    "grad",
    "tversky",
    "height_boost",
    "presence_bce",
    "presence_tversky",
    "aux_height_building",
    "aux_height_vegetation",
)

WEIGHTED_COMPONENTS = (
    "weighted_mae",
    "weighted_ssim",
    "weighted_grad",
    "weighted_tversky",
    "weighted_height_boost",
    "weighted_presence_bce",
    "weighted_presence_tversky",
    "weighted_aux_height",
)

UW_COMPONENTS = (
    "uw_log_var_presence",
    "uw_log_var_fraction",
    "uw_log_var_height",
    "uw_L_presence",
    "uw_L_fraction",
    "uw_L_height",
)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unwrapped_state_dict(model):
    """Unwrap torch.compile's _orig_mod so checkpoints load cleanly into a
    plain (uncompiled) model at inference time."""
    inner = getattr(model, "_orig_mod", model)
    return inner.state_dict()


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type",           default=DEFAULTS["model_type"], choices=MODEL_CHOICES)
    p.add_argument("--output-dir",           default=DEFAULTS["output_dir"])
    p.add_argument("--train-embeddings-dir", default=DEFAULT_TRAIN_EMB)
    p.add_argument("--secondary-train-embeddings-dir", default=None,
                   help="Optional second pixel-aligned embedding dir to concatenate with "
                        "--train-embeddings-dir, e.g. Tessera with AlphaEarth.")
    p.add_argument("--tessera-presence-ch", type=int, default=16,
                   help="Compressed Tessera channels exposed to the residual presence head.")
    p.add_argument("--tessera-hidden-ch", type=int, default=None,
                   help="Hidden width inside the Tessera compressor. Default derives from "
                        "--tessera-presence-ch and preserves the original architecture.")
    p.add_argument("--tessera-hidden-depth", type=int, default=0,
                   help="Extra hidden 3x3 blocks inside the Tessera compressor. Increases "
                        "parameters without changing --tessera-presence-ch.")
    p.add_argument("--train-targets-dir",    default=DEFAULT_TRAIN_TAR)
    p.add_argument("--experiment-name",      default=DEFAULTS["experiment_name"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--patch-size",     type=int,   default=DEFAULTS["patch_size"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay",   type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--grad-accum-steps", type=int, default=DEFAULTS["grad_accum"],
                   help="Accumulate gradients over N mini-batches before optimizer step.")
    p.add_argument("--num-workers",    type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--prefetch-factor", type=int, default=DEFAULTS["prefetch_factor"],
                   help="Batches prefetched per DataLoader worker when num_workers > 0.")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction,
                   default=DEFAULTS["compile"],
                   help="Master 'go fast on A100/H100' switch. When on, applies: "
                        "(1) torch.compile in max-autotune-no-cudagraphs mode — "
                        "Triton kernel autotuning without the CUDA-graph retracing "
                        "pathology at train<->val boundaries; "
                        "(2) persistent Inductor cache at ~/.cache/embed2heights_inductor "
                        "so the ~2-5min warmup is paid once per (code, GPU) pair; "
                        "(3) dynamic=None + mark_dynamic on batch axis so one graph "
                        "handles both full and ragged final batches — no mid-epoch "
                        "recompiles, no dropped samples; "
                        "(4) channels_last memory layout for conv throughput; "
                        "(5) CUDA AMP in bf16 on A100/H100 (fp16 fallback on older "
                        "GPUs) — same dynamic range as fp32, no GradScaler headaches.")
    p.add_argument("--profile-steps", type=int, default=0,
                   help="If >0, time data/forward/backward separately for this many "
                        "steps, print breakdown, and exit.")
    p.add_argument("--deterministic", action="store_true",
                   help="Submission-safe mode: seeds workers, ordered DataLoader, "
                        "deterministic cuDNN, TF32 off. Slower but reproducible.")
    p.add_argument("--aux-weight",     type=float, default=DEFAULTS["aux_weight"],
                   help="Weight for auxiliary multi-head supervision.")
    p.add_argument("--loss-preset", default=DEFAULTS["loss_preset"], choices=LOSS_PRESET_CHOICES,
                   help=(
                       "Loss recipe. auto = presence_centered for tessera_iou_fusion and "
                       "current otherwise; current = existing loss; no_ssim_grad = current "
                       "with SSIM/gradient weights zeroed; presence_centered = presence BCE "
                       "+ presence Tversky + height losses + weak fraction MAE."
                   ))
    p.add_argument("--presence-tversky-weight", type=float,
                   default=DEFAULTS["presence_tversky_weight"],
                   help="Weight for presence Tversky in --loss-preset presence_centered.")
    p.add_argument("--fraction-mae-weight", type=float,
                   default=DEFAULTS["fraction_mae_weight"],
                   help="Weak fraction MAE weight in --loss-preset presence_centered.")
    p.add_argument("--height-loss-kind", default="l1",
                   choices=["l1", "huber", "mse"],
                   help="Regression loss used for height_boost and aux class-height "
                        "supervision. Default l1 matches legacy behavior.")
    p.add_argument("--huber-delta", type=float, default=1.0,
                   help="Transition point for --height-loss-kind huber.")
    p.add_argument("--build-height-boost", type=float, default=5.0,
                   help="Extra per-pixel weight on building-positive pixels inside "
                        "the height_boost term. 5.0 = legacy (previously hardcoded).")
    p.add_argument("--veg-height-boost", type=float, default=0.0,
                   help="Extra per-pixel weight on vegetation-positive pixels inside "
                        "the height_boost term. 0.0 = legacy (no veg boost).")
    p.add_argument("--aux-veg-weight", type=float, default=1.0,
                   help="Multiplier on aux_height_vegetation_loss only. 1.0 = legacy. "
                        "Values > 1 amplify the vegetation specialist's direct L1 "
                        "training signal without touching the shared trunk or the "
                        "mixed-height loss — isolates veg-side learning from the "
                        "presence classifier and building specialist.")
    p.add_argument("--iou-loss-kind", default="tversky",
                   choices=["tversky", "focal"],
                   help="Auxiliary IoU loss form for the presence head under "
                        "--loss-preset presence_centered. 'focal' replaces the "
                        "per-class Tversky with sigmoid focal BCE.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focusing parameter for --iou-loss-kind focal.")
    p.add_argument("--focal-alpha", type=float, default=0.25,
                   help="Positive-class weight for --iou-loss-kind focal.")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="On-the-fly augmentation at train time; subgroup controlled "
                        "by --augment-mode. Val is never augmented.")
    p.add_argument("--augment-mode", default="d4", choices=AUGMENT_MODES,
                   help="Augmentation subgroup. 'd4' = full dihedral (8 variants, "
                        "includes rot90 — can hurt when features encode N-S / "
                        "sun-angle priors). 'flip_rot180' = {id, 180, hflip, "
                        "hflip+180} (4 variants, orientation-preserving). "
                        "'hflip' = horizontal flip only (2 variants). 'none' = "
                        "identity. Ignored when --no-augment.")
    p.add_argument("--aux-tversky-weight", type=float, default=0.0,
                   help="Weight on the auxiliary per-class Tversky loss over the "
                        "main fraction output. Under --loss-preset presence_centered "
                        "this term is off by default (0.0); enabling it adds a "
                        "direct IoU surrogate on the submitted channels alongside "
                        "the presence-head supervision.")
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=False,
                   help="Maintain a step-wise EMA of model parameters; evaluate "
                        "and checkpoint on EMA weights. Usually buys ~1pt IoU at "
                        "zero training cost.")
    p.add_argument("--ema-decay", type=float, default=0.9995,
                   help="EMA decay per optimizer step. Default 0.9995 is a "
                        "reasonable setting for ~60-epoch runs at bs=32.")
    p.add_argument("--scheduler", default="plateau",
                   choices=["plateau", "cosine"],
                   help="LR schedule. plateau = ReduceLROnPlateau(factor=0.5, "
                        "patience=2). cosine = CosineAnnealingLR over --epochs.")
    p.add_argument("--lightunet-base-ch", type=int, default=32,
                   help="Base channel width of LightUNet (also used inside "
                        "tessera_iou_fusion). Decoder channels scale as "
                        "(b, 2b, 4b, 8b). Default 32 = legacy.")
    p.add_argument("--height-specialist-depth", type=int, default=0,
                   help="Extra ConvGNAct layers prepended to the per-class height "
                        "specialist projections (building/vegetation). 0 = legacy 1x1 "
                        "projection only.")
    p.add_argument("--fusion-mode", default="residual_presence",
                   choices=["residual_presence", "gated_feature"],
                   help="How Tessera fuses with AlphaEarth in tessera_iou_fusion. "
                        "residual_presence (legacy) = Tessera adds a zero-init "
                        "residual to presence logits only. gated_feature = "
                        "Tessera is promoted to a peer feature stream and "
                        "fuses into trunk features via a learned spatial gate "
                        "(zero-init weights + +bias so it starts as alpha-only "
                        "and learns Tessera contribution as a residual).")
    p.add_argument("--uncertainty-weighting", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Replace hand-tuned aux/structure/presence_tversky weights "
                        "with learned per-task log-variances (Kendall et al. 2018). "
                        "Three task groups: presence, fraction, height. The "
                        "log-vars are nn.Parameters trained alongside the model.")
    p.add_argument("--structure-weight", type=float, default=None,
                   help="Override lambdas[3] (the weight on height_boost, and Tversky "
                        "under the 'current' preset). Defaults to DEFAULTS['lambdas'][3] "
                        "when unset. Useful when switching height loss kind (e.g. MSE "
                        "changes the magnitude/gradient profile of height_boost).")
    p.add_argument("--seed",           type=int, default=DEFAULTS["seed"])
    p.add_argument("--split-file",     default=None,
                   help="Path to a JSON split file. Loaded if present, else a new split is saved there.")
    return p.parse_args()


def effective_loss_lambdas(args):
    """Return the lambda vector actually used by the selected loss preset."""
    selected = resolve_loss_preset(args)
    structure_w = (args.structure_weight
                   if args.structure_weight is not None
                   else DEFAULTS["lambdas"][3])
    if selected == "no_ssim_grad":
        return [DEFAULTS["lambdas"][0], 0.0, 0.0, structure_w]
    if selected == "presence_centered":
        # Structure weight is still used for height_boost. Fraction Tversky,
        # SSIM, and gradient are disabled inside ImprovedCompositeLoss.
        return [DEFAULTS["lambdas"][0], 0.0, 0.0, structure_w]
    return [DEFAULTS["lambdas"][0], DEFAULTS["lambdas"][1],
            DEFAULTS["lambdas"][2], structure_w]


def resolve_loss_preset(args):
    if args.loss_preset != "auto":
        return args.loss_preset
    if args.model_type.lower() == "tessera_iou_fusion":
        return "presence_centered"
    return "current"


def make_dataloaders(args, device):
    if args.secondary_train_embeddings_dir:
        all_pairs = find_multisource_file_pairs(
            args.train_embeddings_dir,
            args.secondary_train_embeddings_dir,
            args.train_targets_dir,
        )
    else:
        all_pairs = find_file_pairs(args.train_embeddings_dir, args.train_targets_dir)
    if not all_pairs:
        raise ValueError(
            f"No (embedding, label) pairs found.\n"
            f"  train_embeddings_dir='{args.train_embeddings_dir}'\n"
            f"  secondary_train_embeddings_dir='{args.secondary_train_embeddings_dir}'\n"
            f"  train_targets_dir='{args.train_targets_dir}'\n"
            "Check filename conventions and directory paths."
        )

    if args.split_file and os.path.exists(args.split_file):
        train_pairs, val_pairs = load_split(args.split_file, all_pairs)
    else:
        train_pairs, val_pairs = train_test_split(
            all_pairs, test_size=DEFAULTS["val_split"], random_state=args.seed
        )
        if args.split_file:
            save_split(args.split_file, train_pairs, val_pairs)

    with rasterio.open(train_pairs[0][0]) as src:
        n_channels = src.count
    if args.secondary_train_embeddings_dir:
        with rasterio.open(train_pairs[0][1]) as src:
            n_channels += src.count

    DatasetCls = MultiPixelEmbeddingDataset if args.secondary_train_embeddings_dir else pick_dataset_class(args.model_type, n_channels)
    if DatasetCls.__name__ == "LatentTokenDataset":
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size, scale_factor=16,
                              is_train=True, augment=args.augment,
                              augment_mode=args.augment_mode)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size, scale_factor=16,
                              is_train=False, augment=False)
    else:
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size,
                              is_train=True, augment=args.augment,
                              augment_mode=args.augment_mode)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size,
                              is_train=False, augment=False)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    # in_order=False lets faster workers deliver batches ahead of slower ones
    # (e.g. when NFS page-cache state varies across samples). Train only —
    # val stays ordered so eval metrics are deterministic.
    train_in_order = bool(getattr(args, "deterministic", False))
    train_loader_kwargs = dict(loader_kwargs)
    if getattr(args, "deterministic", False):
        # Seed each worker so its RNG state is reproducible across runs.
        def _seed_worker(worker_id):
            import random as _r
            worker_seed = (args.seed + worker_id) % (2 ** 32)
            np.random.seed(worker_seed)
            _r.seed(worker_seed)
        train_loader_kwargs["worker_init_fn"] = _seed_worker
        loader_kwargs["worker_init_fn"] = _seed_worker
    train_loader = DataLoader(train_ds, shuffle=True, in_order=train_in_order,
                              **train_loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_ds, val_ds, n_channels


def forward_for_training(model, imgs):
    if getattr(model, "supports_aux_outputs", False):
        return model(imgs, return_aux=True)
    return model(imgs)


def save_experiment_config(exp_dir, args, device, use_amp):
    os.makedirs(exp_dir, exist_ok=True)
    resolved_loss_preset = resolve_loss_preset(args)
    loss_lambdas = effective_loss_lambdas(args)
    cfg = {
        "experiment_name": args.experiment_name,
        "model_type":      args.model_type,
        "output_dir":      args.output_dir,
        "batch_size":      args.batch_size,
        "patch_size":      args.patch_size,
        "epochs":          args.epochs,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "loss_preset":     resolved_loss_preset,
        "requested_loss_preset": args.loss_preset,
        "loss_lambdas":    loss_lambdas,
        "aux_weight":      args.aux_weight,
        "presence_tversky_weight": args.presence_tversky_weight,
        "fraction_mae_weight": args.fraction_mae_weight,
        "amp":             use_amp,
        "grad_accum":      args.grad_accum_steps,
        "num_workers":     args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "pin_memory":      device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "non_blocking_transfer": device.type == "cuda",
        "seed":            args.seed,
        "train_embeddings_dir": args.train_embeddings_dir,
        "secondary_train_embeddings_dir": args.secondary_train_embeddings_dir,
        "tessera_presence_ch": args.tessera_presence_ch,
        "tessera_hidden_ch":   args.tessera_hidden_ch,
        "tessera_hidden_depth": args.tessera_hidden_depth,
        "height_specialist_depth": args.height_specialist_depth,
        "lightunet_base_ch":   args.lightunet_base_ch,
        "augment":             args.augment,
        "augment_mode":        args.augment_mode if args.augment else "none",
        "aux_tversky_weight":  args.aux_tversky_weight,
        "ema":                 args.ema,
        "ema_decay":           args.ema_decay if args.ema else None,
        "fusion_mode":         args.fusion_mode,
        "uncertainty_weighting": args.uncertainty_weighting,
        "lr_scheduler":        args.scheduler,
        "compile":             args.compile,
        "height_loss_kind":    args.height_loss_kind,
        "huber_delta":         args.huber_delta,
        "build_height_boost":  args.build_height_boost,
        "veg_height_boost":    args.veg_height_boost,
        "aux_veg_weight":      args.aux_veg_weight,
        "iou_loss_kind":       args.iou_loss_kind,
        "focal_gamma":         args.focal_gamma,
        "focal_alpha":         args.focal_alpha,
        "train_targets_dir":    args.train_targets_dir,
        "val_split":       DEFAULTS["val_split"],
        "device":          str(device),
        "optimizer":       "AdamW",
        "scheduler":       "ReduceLROnPlateau(factor=0.5, patience=2)",
        "grad_clip":       1.0,
    }
    with open(os.path.join(exp_dir, "training_params.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Created experiment folder: {exp_dir}")


def _print_profile_summary(prof, n):
    """Pretty-print the per-phase timing collected in run_epoch."""
    import statistics
    print("\n========== PROFILE SUMMARY ==========")
    print(f"Steps measured: {n}")
    print(f"{'phase':<12} {'mean (ms)':>12} {'median (ms)':>14} {'p90 (ms)':>12} {'% of total':>12}")
    total_mean = statistics.mean(prof["total"]) if prof["total"] else 1.0
    for phase in ("data", "h2d", "forward", "backward", "optimizer", "total"):
        vals = prof[phase]
        if not vals:
            continue
        mean_ms = statistics.mean(vals) * 1000
        med_ms = statistics.median(vals) * 1000
        p90_ms = sorted(vals)[int(0.9 * (len(vals) - 1))] * 1000
        pct = 100 * statistics.mean(vals) / total_mean if phase != "total" else 100.0
        print(f"{phase:<12} {mean_ms:>12.1f} {med_ms:>14.1f} {p90_ms:>12.1f} {pct:>11.1f}%")
    print("=====================================")


def run_epoch(model, loader, criterion, optimizer, scaler, device, *, train,
              grad_accum_steps=1, use_amp=False, amp_dtype=torch.float16,
              channels_last=False, profile_steps=0, desc="", ema=None):
    """Train or eval one epoch. Returns (avg_loss, component_avgs)."""
    model.train(train)
    running_loss_t = torch.zeros((), device=device)
    component_sums_t = {}
    samples_seen = 0
    use_scaler = scaler is not None and scaler.is_enabled()
    prof = {"data": [], "h2d": [], "forward": [], "backward": [], "optimizer": [], "total": []}
    sync_cuda = (device.type == "cuda") and profile_steps > 0

    pbar = tqdm(loader, desc=desc, leave=False)
    if train:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if train else torch.no_grad()
    non_blocking = device.type == "cuda"
    with context:
        loader_iter = iter(pbar)
        step = 0
        while True:
            step += 1
            t_step_start = time.perf_counter()
            try:
                imgs, targets, masks = next(loader_iter)
            except StopIteration:
                step -= 1
                if profile_steps > 0 and prof["total"]:
                    _print_profile_summary(prof, len(prof["total"]))
                    import sys; sys.exit(0)
                break
            t_after_data = time.perf_counter()
            imgs = imgs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            masks = masks.to(device, non_blocking=non_blocking)
            if channels_last:
                imgs = imgs.contiguous(memory_format=torch.channels_last)
            # Pre-declare batch dim (axis 0) as dynamic on the first step of a
            # compiled run. Lets Inductor emit one graph that handles both the
            # full batch and the ragged final batch, avoiding a mid-epoch
            # recompile. No-op when the model isn't wrapped by torch.compile.
            if step == 1 and hasattr(model, "_orig_mod"):
                torch._dynamo.mark_dynamic(imgs, 0)
                torch._dynamo.mark_dynamic(targets, 0)
                torch._dynamo.mark_dynamic(masks, 0)
            if sync_cuda:
                torch.cuda.synchronize()
            t_after_h2d = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = forward_for_training(model, imgs)
                loss, loss_components = criterion(outputs, targets, masks)
                step_loss = loss / grad_accum_steps if train else loss
            if sync_cuda:
                torch.cuda.synchronize()
            t_after_fwd = time.perf_counter()

            if not torch.isfinite(loss):
                print(f"\nWARN: non-finite loss at step {step}; skipping batch.")
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train:
                if use_scaler:
                    scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()
                if sync_cuda:
                    torch.cuda.synchronize()
            t_after_bwd = time.perf_counter()

            if train:
                should_step = step % grad_accum_steps == 0 or step == len(loader)
                if should_step:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    # Clip across every param group the optimizer owns —
                    # this includes criterion log-vars under uncertainty
                    # weighting.
                    clip_targets = [p for g in optimizer.param_groups for p in g["params"]]
                    torch.nn.utils.clip_grad_norm_(clip_targets, max_norm=1.0)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)
            if sync_cuda:
                torch.cuda.synchronize()
            t_after_opt = time.perf_counter()

            if profile_steps > 0:
                prof["data"].append(t_after_data - t_step_start)
                prof["h2d"].append(t_after_h2d - t_after_data)
                prof["forward"].append(t_after_fwd - t_after_h2d)
                prof["backward"].append(t_after_bwd - t_after_fwd)
                prof["optimizer"].append(t_after_opt - t_after_bwd)
                prof["total"].append(t_after_opt - t_step_start)
                if step >= profile_steps:
                    _print_profile_summary(prof, profile_steps)
                    import sys; sys.exit(0)

            bs = imgs.size(0)
            running_loss_t += loss.detach() * bs
            for name, value in loss_components.items():
                if name not in component_sums_t:
                    component_sums_t[name] = torch.zeros((), device=device)
                component_sums_t[name] += value.detach() * bs
            samples_seen += bs
            if step % 20 == 0 or step == len(loader):
                avg = (running_loss_t / max(1, samples_seen)).item()
                pbar.set_postfix(avg=f"{avg:.4f}")

    avg_loss = (running_loss_t / max(1, samples_seen)).item()
    comp_avg = {
        name: (value / max(1, samples_seen)).item()
        for name, value in component_sums_t.items()
    }
    return avg_loss, comp_avg


def format_components(components, names):
    parts = []
    for name in names:
        if name in components:
            parts.append(f"{name}:{components[name]:.3f}")
    return " | ".join(parts)


def write_history_record(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def plot_loss_curve(train_losses, val_losses, out_path, experiment_name):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN: matplotlib unavailable; skipping loss curve: {exc}")
        return

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Validation Loss")
    plt.title(f"Training Loss Curve ({experiment_name})")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    device = select_device()
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    best_model_path = os.path.join(exp_dir, "model_best.pth")
    last_model_path = os.path.join(exp_dir, "model_last.pth")
    loss_curve_path = os.path.join(exp_dir, "loss_curve.png")
    loss_history_path = os.path.join(exp_dir, "loss_history.jsonl")

    # AMP is part of the --compile "go fast" bundle on CUDA.
    use_amp = args.compile and device.type == "cuda"
    # bf16 is the right AMP dtype on A100 (same dynamic range as fp32, no GradScaler headaches).
    # Fall back to fp16 on older GPUs that don't support bf16.
    amp_dtype = torch.float16
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    if device.type == "cuda":
        if args.deterministic:
            # Submission-safe: cuDNN chooses a fixed (slower) algo each call,
            # TF32 off so matmuls stay bit-compatible, autograd flags a warning
            # on any op that has no deterministic implementation.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.cuda.manual_seed_all(args.seed)
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    # channels_last is part of the --compile "go fast" bundle on CUDA.
    channels_last = args.compile and device.type == "cuda"
    grad_accum_steps = max(1, args.grad_accum_steps)
    save_experiment_config(exp_dir, args, device, use_amp)
    open(loss_history_path, "w").close()

    print("--- 1. Data Setup ---")
    train_loader, val_loader, train_ds, val_ds, n_channels = make_dataloaders(args, device)

    print("--- 2. Model Init ---")
    model, selected_model = build_model(
        args.model_type,
        n_channels,
        n_classes=4,
        tessera_presence_ch=args.tessera_presence_ch,
        tessera_hidden_ch=args.tessera_hidden_ch,
        tessera_hidden_depth=args.tessera_hidden_depth,
        height_specialist_depth=args.height_specialist_depth,
        lightunet_base_ch=args.lightunet_base_ch,
        fusion_mode=args.fusion_mode,
    )
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile and device.type == "cuda" and hasattr(torch, "compile"):
        # Persist Inductor's compiled kernels across SLURM jobs so we pay the
        # warmup once per (code, hardware) pair, not once per job.
        cache_dir = os.path.expanduser("~/.cache/embed2heights_inductor")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
        # max-autotune-no-cudagraphs: Triton kernel autotuning without CUDA
        # graphs. CUDA-graph modes ('reduce-overhead', 'max-autotune') retrace
        # at the train<->val boundary because val skips augmentation/dropout,
        # which changes shapes/strides — this is the only autotune mode that
        # avoids that pathology.
        # dynamic=None: first shape compiles static (full optimizations),
        # ragged final batch triggers a one-time fallback to a batch-dim-dynamic
        # graph. Both graphs cached, zero recompiles from epoch 2 onward.
        model = torch.compile(model, mode="max-autotune-no-cudagraphs",
                              dynamic=None)
        print(f"torch.compile: mode=max-autotune-no-cudagraphs, dynamic=None, "
              f"cache={os.environ['TORCHINDUCTOR_CACHE_DIR']}")
    print(f"Using model: {selected_model} (input channels={n_channels})")

    fused_ok = device.type == "cuda"
    # GradScaler is only meaningful for fp16; bf16 and fp32 don't need it.
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    resolved_loss_preset = resolve_loss_preset(args)
    loss_lambdas = effective_loss_lambdas(args)
    criterion = ImprovedCompositeLoss(
        lambdas=loss_lambdas,
        aux_weight=args.aux_weight,
        loss_preset=resolved_loss_preset,
        presence_tversky_weight=args.presence_tversky_weight,
        fraction_mae_weight=args.fraction_mae_weight,
        height_loss_kind=args.height_loss_kind,
        huber_delta=args.huber_delta,
        build_height_boost=args.build_height_boost,
        veg_height_boost=args.veg_height_boost,
        aux_veg_weight=args.aux_veg_weight,
        iou_loss_kind=args.iou_loss_kind,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        aux_tversky_weight=args.aux_tversky_weight,
    ).to(device)
    if args.uncertainty_weighting:
        criterion = UncertaintyWeightedLoss(criterion).to(device)
        print("Uncertainty weighting enabled (3 learned log-variances: presence, fraction, height).")

    # Optimizer must see the criterion's parameters too when uncertainty
    # weighting is on (the log-variances are learned).
    trainable_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr,
                            weight_decay=args.weight_decay, fused=fused_ok)
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
        )
        scheduler_step_on_val = False
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        scheduler_step_on_val = True

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    if ema is not None:
        print(f"EMA enabled (decay={args.ema_decay})")
    print(
        "Using loss: "
        f"preset={resolved_loss_preset} (requested={args.loss_preset}), "
        f"lambdas={loss_lambdas}, "
        f"aux_weight={args.aux_weight}, "
        f"presence_tversky_weight={args.presence_tversky_weight}, "
        f"fraction_mae_weight={args.fraction_mae_weight}, "
        f"height_loss_kind={args.height_loss_kind}, "
        f"huber_delta={args.huber_delta}, "
        f"build_height_boost={args.build_height_boost}, "
        f"veg_height_boost={args.veg_height_boost}, "
        f"aux_veg_weight={args.aux_veg_weight}, "
        f"iou_loss_kind={args.iou_loss_kind}, "
        f"focal_gamma={args.focal_gamma}, "
        f"focal_alpha={args.focal_alpha}"
    )

    print(f"Starting training on {device}...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        tr_loss, tr_comp = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, grad_accum_steps=grad_accum_steps, use_amp=use_amp,
            amp_dtype=amp_dtype, channels_last=channels_last,
            profile_steps=args.profile_steps,
            desc=f"Epoch {epoch + 1}/{args.epochs} [train]",
            ema=ema,
        )
        # Validate on EMA weights when enabled — that's the checkpoint we save.
        if ema is not None:
            ema.store(model)
        val_loss, val_comp = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, use_amp=use_amp,
            amp_dtype=amp_dtype, channels_last=channels_last,
            desc=f"Epoch {epoch + 1}/{args.epochs} [val]",
        )
        if ema is not None:
            ema.restore(model)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        if scheduler_step_on_val:
            scheduler.step(val_loss)
        else:
            scheduler.step()
        write_history_record(loss_history_path, {
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "train_components": tr_comp,
            "val_components": val_comp,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # val_loss was measured on EMA weights (if enabled), so save EMA
            # weights to match — that's the model the leaderboard actually sees.
            save_sd = ema.state_dict() if ema is not None else _unwrapped_state_dict(model)
            torch.save(save_sd, best_model_path)
            print(f"   >> New best val loss {best_val_loss:.4f} — saved.")

        print(f"Epoch {epoch + 1}/{args.epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
        print(f"   >> Train raw: {format_components(tr_comp, RAW_COMPONENTS)}")
        print(f"   >> Train weighted: {format_components(tr_comp, WEIGHTED_COMPONENTS)}")
        print(f"   >> Val raw:   {format_components(val_comp, RAW_COMPONENTS)}")
        print(f"   >> Val weighted: {format_components(val_comp, WEIGHTED_COMPONENTS)}")
        if args.uncertainty_weighting:
            print(f"   >> UW: {format_components(tr_comp, UW_COMPONENTS)}")

    print("--- 3. Saving ---")
    last_sd = ema.state_dict() if ema is not None else _unwrapped_state_dict(model)
    torch.save(last_sd, last_model_path)
    plot_loss_curve(train_losses, val_losses, loss_curve_path, args.experiment_name)


if __name__ == "__main__":
    main()
