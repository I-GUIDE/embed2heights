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
import numpy as np
import torch
import torch.optim as optim
import rasterio
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.models import build_model
from core.data import (
    find_file_pairs, find_multisource_file_pairs, find_trisource_file_pairs,
    find_quadsource_file_pairs,
    save_split, load_split,
)
from core.data.datasets import (
    pick_dataset_class, MultiPixelEmbeddingDataset,
    MultiLatentTokenDataset, PixelTokenEmbeddingDataset, PixelMultiTokenEmbeddingDataset,
)
from core.losses import ImprovedCompositeLoss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/projects/bcrm/emb2height/data"
DEFAULT_TRAIN_EMB = os.path.join(DATA_ROOT, "train", "alphaearth_emb")
DEFAULT_TRAIN_TAR = os.path.join(DATA_ROOT, "train", "labels")

MODEL_CHOICES = [
    "auto", "lightunet", "lightunet_presence_2plus1", "lightunet_presence_3way",
    "lightunet_presence_shared3",
    "lightunet_pp", "decoder_residual", "token_neck",
    "token_neck_norm", "token_fusion_neck", "token_fusion_neck_norm",
    "token_fusion_neck_xattn", "token_fusion_neck_xattn_norm",
    "embedding_refiner", "hrnet_w18", "hrnet_w32", "tessera_iou_fusion",
    "tessera_iou_fusion_unetpp", "tessera_iou_fusion_presence_2plus1",
    "tessera_iou_fusion_presence_3way", "tessera_iou_fusion_presence_shared3",
    "tessera_iou_fusion_gated", "tessera_iou_fusion_gated_presence_2plus1",
    "tessera_iou_fusion_gated_presence_3way",
    "tessera_iou_fusion_gated_presence_shared3", "tessera_token_shared_probe",
    "tessera_token_fusion_shared_probe", "tessera_token_fusion_shared_probe_norm",
    "tessera_token_height_residual_probe", "tessera_token_xattn_height_residual_probe",
    "tessera_token_s2_nonwater_residual_decoder64",
    "tessera_token_s2_all_residual_decoder64",
    "tessera_token_s2_water_residual_decoder64",
    "tessera_token_crosslevel_s2_bottleneck",
    "tessera_token_crosslevel_s2_decoder64",
    "tessera_token_crosslevel_s2_decoder64_presence_2plus1",
    "tessera_token_crosslevel_s2_decoder64_presence_3way",
    "tessera_token_crosslevel_s2_decoder64_presence_3way_deep",
    "tessera_token_crosslevel_s2_bottleneck_decoder64_presence_3way_deep",
    "tessera_token_crosslevel_s2_decoder64_decoder128_presence_3way_deep",
    "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_water_bypass",
    "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated",
    "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated_tessera_gated",
    "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated_hada_tessera_gated",
    "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated",
    "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated_norm",
    "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated_tessera_gated",
    "tessera_token_crosslevel_s2_dpt_film_3way_deep_norm",
    "dpt_compact_token_only", "dpt_compact_token_only_3way_deep",
    "tessera_token_crosslevel_xattn_bottleneck",
    "tessera_token_crosslevel_xattn_decoder64",
    "tessera_token_crosslevel_xattn_bottleneck_decoder64",
    # Active strategy aliases (registry.py)
    "ae_only", "ae_tessera_gated", "xfusion_crosslevel",
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
    "amp":             True,
    "grad_accum":      1,
    "num_workers":     4,
    "prefetch_factor": 1,
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
    "height_bin_ce",
    "building_smooth",
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
    "weighted_height_bin_ce",
    "weighted_building_smooth",
)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    p.add_argument("--token-train-embeddings-dir", default=None,
                   help="Optional 16x16 token embedding dir for shared-probe fusion with "
                        "AlphaEarth+Tessera. Requires --secondary-train-embeddings-dir.")
    p.add_argument("--secondary-token-train-embeddings-dir", default=None,
                   help="Optional second 16x16 token embedding dir for same-model S1/S2 "
                        "shared-probe fusion. Requires --token-train-embeddings-dir and "
                        "--secondary-train-embeddings-dir.")
    p.add_argument("--tessera-presence-ch", type=int, default=16,
                   help="Compressed Tessera channels exposed to the residual presence head.")
    p.add_argument("--tessera-hidden-ch", type=int, default=None,
                   help="Hidden width inside the Tessera compressor. Default derives from "
                        "--tessera-presence-ch and preserves the original architecture.")
    p.add_argument("--tessera-hidden-depth", type=int, default=0,
                   help="Extra hidden 3x3 blocks inside the Tessera compressor. Increases "
                        "parameters without changing --tessera-presence-ch.")
    p.add_argument("--gate-mode", choices=["simple", "rich"], default="simple",
                   help="Fusion gate variant for tessera_iou_fusion_gated: simple = 1x1 "
                        "conv, rich = Conv1x1 -> GN -> GELU -> Conv1x1.")
    p.add_argument("--gate-untied", action="store_true",
                   help="Use untied gates (G_AE and G_TES independent) instead of tied "
                        "G/(1-G). Strictly more expressive (GMU Figure 2a).")
    p.add_argument("--gate-init-bias", type=float, default=4.0,
                   help="Bias init for the fusion gate so sigmoid(b) ~ 1 at step 0 "
                        "(warm-start as AlphaEarth-only).")
    p.add_argument("--modality-dropout", type=float, default=0.0,
                   help="Per-sample probability of zeroing the Tessera feature stream "
                        "during training. Inverted-dropout scaling. Keeps the AE branch "
                        "self-sufficient.")
    p.add_argument("--train-targets-dir",    default=DEFAULT_TRAIN_TAR)
    p.add_argument("--experiment-name",      default=DEFAULTS["experiment_name"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--patch-size",     type=int,   default=DEFAULTS["patch_size"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay",   type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="Apply torch.compile(mode='default') + channels_last + high matmul "
                        "precision. Balanced speedup (~20-30%%) without the long kernel search "
                        "of max-autotune. Requires PyTorch >= 2.0.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=DEFAULTS["amp"],
                   help="Use CUDA automatic mixed precision when available.")
    p.add_argument("--data-parallel", action="store_true",
                   help="Wrap the model in torch.nn.DataParallel when multiple CUDA "
                        "devices are visible. This splits each batch across GPUs.")
    p.add_argument("--grad-accum-steps", type=int, default=DEFAULTS["grad_accum"],
                   help="Accumulate gradients over N mini-batches before optimizer step.")
    p.add_argument("--num-workers",    type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--prefetch-factor", type=int, default=DEFAULTS["prefetch_factor"],
                   help="Batches prefetched per DataLoader worker when num_workers > 0.")
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
    p.add_argument("--lightunet-base-ch", type=int, default=32,
                   help="Base channel width of LightUNet (also used inside "
                        "tessera_iou_fusion). Decoder channels scale as "
                        "(b, 2b, 4b, 8b). Default 32 = legacy.")
    p.add_argument("--lightunet-norm-kind", default="bn", choices=["bn", "gn"],
                   help="Normalization layer inside LightUNet's DoubleConv / "
                        "UpsampleBlock. 'bn' = legacy BatchNorm. 'gn' = "
                        "GroupNorm (stateless), required when initialising from "
                        "an SSL pretrain checkpoint that was produced with GN — "
                        "BN running stats do not transfer cleanly from "
                        "label-free pretraining to the supervised regime.")
    p.add_argument("--height-specialist-depth", type=int, default=0,
                   help="Extra ConvGNAct layers prepended to the per-class height "
                        "specialist projections (building/vegetation). 0 = legacy 1x1 "
                        "projection only.")
    p.add_argument("--height-gate-source", default="alpha",
                   choices=["alpha", "fused"],
                   help="Presence logits used to route the submitted height. alpha = "
                        "legacy AlphaEarth-only logits; fused = AlphaEarth logits plus "
                        "the Tessera residual presence correction.")
    p.add_argument("--height-hidden-ch", type=int, default=None,
                   help="Internal channel width for the height trunk. Default matches "
                        "the shared head width and preserves legacy behavior.")
    p.add_argument("--height-trunk-depth", type=int, default=2,
                   help="Number of ConvGNAct blocks in the height trunk. Default 2 "
                        "matches legacy behavior.")
    p.add_argument("--height-independent-branches", action="store_true",
                   help="Use separate base/building/vegetation height trunks instead "
                        "of one shared height_trunk feeding all projections.")
    p.add_argument("--height-head-kind", default="linear", choices=["linear", "softbin"],
                   help="Output parameterization for the per-class height heads. "
                        "'linear' = legacy softplus regression on a single channel; "
                        "'softbin' = AdaBins-style soft-classification over K log-spaced "
                        "bin centers, with expectation as the predicted height. The "
                        "soft-bin head pairs with --height-bin-aux-weight for the "
                        "Gaussian-soft-target CE that prevents regression-to-mean.")
    p.add_argument("--height-n-bins", type=int, default=64,
                   help="Number of bins for --height-head-kind softbin. Bin centers "
                        "are log1p-spaced over [0, --height-bin-max-m] meters.")
    p.add_argument("--height-bin-max-m", type=float, default=80.0,
                   help="Upper bin edge (in meters) for --height-head-kind softbin. "
                        "Should comfortably exceed observed heights so the model can "
                        "place mass on tall pixels without saturating.")
    p.add_argument("--height-bin-aux-weight", type=float, default=0.0,
                   help="Weight for the soft-bin auxiliary cross-entropy on log-height. "
                        "0.0 disables it. Only meaningful with --height-head-kind softbin.")
    p.add_argument("--height-bin-sigma-bins", type=float, default=1.5,
                   help="Width (in bin units) of the Gaussian soft target used by the "
                        "bin CE auxiliary. Smaller = sharper / more committal targets.")
    p.add_argument("--building-smooth-weight", type=float, default=0.0,
                   help="Weight for height total-variation loss inside eroded GT "
                        "building interiors. 0.0 disables it.")
    p.add_argument("--building-smooth-erode-px", type=int, default=1,
                   help="Binary erosion radius in pixels before applying building "
                        "interior smoothness. Keeps the loss off building boundaries.")
    p.add_argument("--building-smooth-thr", type=float, default=0.0,
                   help="GT building fraction threshold used to define building "
                        "pixels for interior smoothness.")
    p.add_argument("--lds-sampler", action="store_true",
                   help="Replace shuffle=True on the train DataLoader with a "
                        "WeightedRandomSampler whose per-patch weights are 1/n_b on "
                        "an LDS-smoothed (Yang et al. ICML 2021) histogram of a "
                        "patch tail/density score. The val loader is unchanged "
                        "(natural distribution).")
    p.add_argument("--lds-bins", type=int, default=16,
                   help="Number of bins over [0, --lds-h-max] for LDS scoring.")
    p.add_argument("--lds-sigma", type=float, default=2.0,
                   help="Gaussian smoothing kernel σ (in bin units) for LDS. "
                        "Larger σ = smoother weights = less aggressive resampling.")
    p.add_argument("--lds-cap", type=float, default=5.0,
                   help="Cap on max(weight) / median(weight). Prevents one rare bin "
                        "from dominating the sampler when σ is small.")
    p.add_argument("--lds-h-max", type=float, default=60.0,
                   help="Upper edge of the score histogram. Patches with p95 height "
                        "above this are clipped into the top bin.")
    p.add_argument("--lds-score", default="p95_veg",
                   choices=[
                       "p95_veg", "p99_veg", "max_veg",
                       "p95_building", "p99_building", "max_building",
                       "building_frac_p95",
                   ],
                   help="Per-patch tail-severity score driving the LDS sampler. "
                        "p95_veg = legacy (95th pct of vegetation height). p99_veg "
                        "and max_veg make the sampler far more sensitive to patches "
                        "that hide a few extreme tall pixels — empirically p95 missed "
                        "the 40m+ tail entirely on this dataset (max patch p95 was "
                        "only 31.3m). p95_building / p99_building / max_building "
                        "target building-height tails. building_frac_p95 combines "
                        "building p95 with building pixel density for KE-like dense "
                        "tall-building regions.")
    p.add_argument("--task", default="both", choices=["both", "presence", "height"],
                   help="Single-task split control. 'both' = legacy multi-task training "
                        "(every loss term active). 'presence' = train only the IoU/"
                        "presence side; height losses are zeroed in total_loss. "
                        "'height' = train only the height regression side; presence "
                        "losses are zeroed. Use this for the dual-model split where "
                        "one run owns submission channels 0-2 and the other owns "
                        "channel 3.")
    p.add_argument("--structure-weight", type=float, default=None,
                   help="Override lambdas[3] (the weight on height_boost, and Tversky "
                        "under the 'current' preset). Defaults to DEFAULTS['lambdas'][3] "
                        "when unset. Useful when switching height loss kind (e.g. MSE "
                        "changes the magnitude/gradient profile of height_boost).")
    p.add_argument("--seed",           type=int, default=DEFAULTS["seed"])
    p.add_argument("--split-file",     default=None,
                   help="Path to a JSON split file. Loaded if present, else a new split is saved there.")
    p.add_argument("--init-from-pretrain", default=None,
                   help="Optional self-supervised pretrain checkpoint. Compatible "
                        "weights are loaded by exact key and shape match before "
                        "supervised training starts; omitted by default.")
    p.add_argument("--init-pretrain-strict", action="store_true",
                   help="Use strict checkpoint loading for --init-from-pretrain. "
                        "Default is non-strict filtered loading, which ignores "
                        "pretraining-only reconstruction heads.")
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
    if args.model_type.lower() in {
        "lightunet_presence_2plus1", "lightunet_presence_3way",
        "lightunet_presence_shared3",
        "tessera_iou_fusion", "tessera_iou_fusion_presence_2plus1",
        "tessera_iou_fusion_presence_3way", "tessera_iou_fusion_presence_shared3",
        "tessera_iou_fusion_gated", "tessera_iou_fusion_gated_presence_2plus1",
        "tessera_iou_fusion_gated_presence_3way",
        "tessera_iou_fusion_gated_presence_shared3", "tessera_token_shared_probe",
        "tessera_token_fusion_shared_probe", "tessera_token_fusion_shared_probe_norm",
        "tessera_token_height_residual_probe", "tessera_token_xattn_height_residual_probe",
        "tessera_token_s2_nonwater_residual_decoder64",
        "tessera_token_s2_all_residual_decoder64",
        "tessera_token_s2_water_residual_decoder64",
        "tessera_token_crosslevel_s2_bottleneck",
        "tessera_token_crosslevel_s2_decoder64",
        "tessera_token_crosslevel_s2_decoder64_presence_2plus1",
        "tessera_token_crosslevel_s2_decoder64_presence_3way",
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep",
        "tessera_token_crosslevel_s2_bottleneck_decoder64_presence_3way_deep",
        "tessera_token_crosslevel_s2_decoder64_decoder128_presence_3way_deep",
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_water_bypass",
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated",
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated_tessera_gated",
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated_hada_tessera_gated",
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated",
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated_norm",
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated_tessera_gated",
        "tessera_token_crosslevel_s2_dpt_film_3way_deep_norm",
        "dpt_compact_token_only",
        "dpt_compact_token_only_3way_deep",
        "tessera_token_crosslevel_xattn_bottleneck",
        "tessera_token_crosslevel_xattn_decoder64",
        "tessera_token_crosslevel_xattn_bottleneck_decoder64",
    }:
        return "presence_centered"
    return "current"


def _patch_tail_score(label, kind):
    """Per-patch tail-severity score; assumes `label` is (4, H, W) float32."""
    valid = ~np.all(label == 0, axis=0)
    any_class = (label[0] + label[1] + label[2]) > 0
    height_hole = (label[3] == 0) & any_class
    height_valid = valid & ~height_hole
    if not height_valid.any():
        return 0.0

    if kind in {"p95_building", "p99_building", "max_building", "building_frac_p95"}:
        class_mask = (label[0] > 0) & height_valid
    else:
        class_mask = (label[1] > 0) & height_valid

    pool = label[3][class_mask] if class_mask.any() else label[3][height_valid]
    if pool.size == 0:
        return 0.0
    if kind in {"p95_veg", "p95_building"}:
        return float(np.percentile(pool, 95))
    if kind in {"p99_veg", "p99_building"}:
        return float(np.percentile(pool, 99))
    if kind in {"max_veg", "max_building"}:
        return float(pool.max())
    if kind == "building_frac_p95":
        # Dense-building domains are the public/fold0 failure mode. Multiplying
        # by a small density factor keeps this mostly a height-tail score while
        # separating KE-like dense tall-building patches from sparse tall outliers.
        building_frac = float(class_mask.sum() / max(1, height_valid.sum()))
        return float(np.percentile(pool, 95) * (1.0 + 4.0 * building_frac))
    raise ValueError(f"unknown lds-score kind: {kind!r}")


def build_lds_sampler(train_pairs, *, n_bins, sigma_bins, cap, h_max, exp_dir,
                      score_kind="p95_veg"):
    """Patch-level WeightedRandomSampler with LDS-smoothed weights.

    Score per patch is selected by `score_kind`. Vegetation variants target
    canopy tails; building variants target building-height tails; the
    building_frac_p95 variant also rewards dense building masks. All variants
    fall back to the same percentile / max over all valid height pixels when a
    patch has no target-class pixels. Bin scores into n_bins over [0, h_max],
    smooth bin frequencies with a Gaussian of σ = sigma_bins (in bin units),
    take inverse, then cap so max(w) / median(w) <= cap.

    Writes <exp_dir>/lds_sampler_weights.json with score + bin-count + weight
    diagnostics for inspection.
    """
    import time
    print(f"LDS sampler: scoring {len(train_pairs)} train patches by {score_kind} ...")
    t0 = time.time()
    scores = np.zeros(len(train_pairs), dtype=np.float32)
    for i, pair in enumerate(train_pairs):
        label_path = pair[-1]
        if label_path is None:
            continue
        with rasterio.open(label_path) as src:
            label = src.read().astype(np.float32)
        scores[i] = _patch_tail_score(label, score_kind)
        if (i + 1) % 200 == 0 or (i + 1) == len(train_pairs):
            print(f"  scored {i+1}/{len(train_pairs)} patches  t={time.time()-t0:.1f}s")

    edges = np.linspace(0.0, h_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)

    # Kernel half-width capped so the kernel never exceeds the bin grid;
    # otherwise np.convolve(..., mode='same') returns max(len(counts), len(kernel))
    # and bin_weight ends up longer than n_bins.
    half = max(1, min(int(np.ceil(3 * sigma_bins)), (n_bins - 1) // 2))
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma_bins ** 2))
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")[:n_bins]
    smooth = np.maximum(smooth, 1.0)  # keep empty bins from going to inf

    bin_weight = 1.0 / smooth
    weights = bin_weight[bin_idx]
    med = float(np.median(weights))
    weights = np.minimum(weights, med * cap)
    weights = weights / weights.mean()

    diag = {
        "n_pairs":    int(len(scores)),
        "score_kind": score_kind,
        "h_max":      float(h_max),
        "n_bins":     int(n_bins),
        "sigma_bins": float(sigma_bins),
        "cap_ratio":  float(cap),
        "score_stats": {
            "min": float(scores.min()),
            "p5":  float(np.percentile(scores, 5)),
            "p50": float(np.percentile(scores, 50)),
            "p95": float(np.percentile(scores, 95)),
            "max": float(scores.max()),
        },
        "bin_edges":           edges.tolist(),
        "bin_counts_raw":      counts.tolist(),
        "bin_counts_smoothed": smooth.tolist(),
        "bin_weight":          bin_weight.tolist(),
        "weight_stats": {
            "min":    float(weights.min()),
            "median": float(np.median(weights)),
            "max":    float(weights.max()),
            "mean":   float(weights.mean()),
        },
    }
    os.makedirs(exp_dir, exist_ok=True)
    out_path = os.path.join(exp_dir, "lds_sampler_weights.json")
    with open(out_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(
        f"LDS sampler: ready in {time.time()-t0:.1f}s  "
        f"weights min/med/max = "
        f"{weights.min():.3f}/{np.median(weights):.3f}/{weights.max():.3f}  "
        f"(diag: {out_path})"
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def make_dataloaders(args, device):
    if args.secondary_token_train_embeddings_dir:
        if not args.token_train_embeddings_dir or not args.secondary_train_embeddings_dir:
            raise ValueError(
                "--secondary-token-train-embeddings-dir requires "
                "--token-train-embeddings-dir and --secondary-train-embeddings-dir"
            )
        all_pairs = find_quadsource_file_pairs(
            args.train_embeddings_dir,
            args.secondary_train_embeddings_dir,
            args.token_train_embeddings_dir,
            args.secondary_token_train_embeddings_dir,
            args.train_targets_dir,
        )
    elif args.token_train_embeddings_dir:
        if not args.secondary_train_embeddings_dir:
            raise ValueError("--token-train-embeddings-dir requires --secondary-train-embeddings-dir")
        all_pairs = find_trisource_file_pairs(
            args.train_embeddings_dir,
            args.secondary_train_embeddings_dir,
            args.token_train_embeddings_dir,
            args.train_targets_dir,
        )
    elif args.secondary_train_embeddings_dir:
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
            f"  token_train_embeddings_dir='{args.token_train_embeddings_dir}'\n"
            f"  secondary_token_train_embeddings_dir='{args.secondary_token_train_embeddings_dir}'\n"
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
    if args.secondary_token_train_embeddings_dir:
        with rasterio.open(train_pairs[0][1]) as src:
            pixel_channels = n_channels + src.count
        with rasterio.open(train_pairs[0][2]) as src:
            token_channels = src.count
        with rasterio.open(train_pairs[0][3]) as src:
            token_channels += src.count
        n_channels = (pixel_channels, token_channels)
        DatasetCls = PixelMultiTokenEmbeddingDataset
    elif args.token_train_embeddings_dir:
        with rasterio.open(train_pairs[0][1]) as src:
            pixel_channels = n_channels + src.count
        with rasterio.open(train_pairs[0][2]) as src:
            token_channels = src.count
        n_channels = (pixel_channels, token_channels)
        DatasetCls = PixelTokenEmbeddingDataset
    else:
        if args.secondary_train_embeddings_dir:
            with rasterio.open(train_pairs[0][1]) as src:
                n_channels += src.count
        if args.secondary_train_embeddings_dir and args.model_type.lower() in {
            "token_fusion_neck", "token_fusion_neck_norm",
            "token_fusion_neck_xattn", "token_fusion_neck_xattn_norm",
        }:
            DatasetCls = MultiLatentTokenDataset
        else:
            DatasetCls = MultiPixelEmbeddingDataset if args.secondary_train_embeddings_dir else pick_dataset_class(args.model_type, n_channels)

    if DatasetCls.__name__ == "LatentTokenDataset":
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size, scale_factor=16, is_train=True)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size, scale_factor=16, is_train=False)
    elif DatasetCls.__name__ in {"PixelTokenEmbeddingDataset", "PixelMultiTokenEmbeddingDataset", "MultiLatentTokenDataset"}:
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size, scale_factor=16, is_train=True)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size, scale_factor=16, is_train=False)
    else:
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size, is_train=True)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size, is_train=False)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    if getattr(args, "lds_sampler", False):
        sampler = build_lds_sampler(
            train_pairs,
            n_bins=args.lds_bins,
            sigma_bins=args.lds_sigma,
            cap=args.lds_cap,
            h_max=args.lds_h_max,
            exp_dir=os.path.join(args.output_dir, args.experiment_name),
            score_kind=args.lds_score,
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_ds, val_ds, n_channels


def move_to_device(batch, device, non_blocking=False):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device, non_blocking=non_blocking) for item in batch)
    if isinstance(batch, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in batch]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def batch_size_of(batch):
    if torch.is_tensor(batch):
        return batch.size(0)
    if isinstance(batch, (tuple, list)) and batch:
        return batch_size_of(batch[0])
    raise TypeError(f"Unsupported batch type for batch size: {type(batch)!r}")


def forward_for_training(model, imgs):
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if getattr(base_model, "supports_aux_outputs", False):
        return model(imgs, return_aux=True)
    return model(imgs)


def state_dict_for_save(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_pretrain_weights(model, checkpoint_path, *, strict=False):
    """Load compatible self-supervised weights without touching unmatched heads."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }

    if strict:
        result = model.load_state_dict(state, strict=True)
        print(f"Loaded strict pretrain checkpoint: {checkpoint_path}")
        return result

    target = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in target and target[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    if not compatible:
        raise ValueError(
            f"No compatible tensors found in pretrain checkpoint: {checkpoint_path}"
        )

    result = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded {len(compatible)} compatible tensors from pretrain checkpoint "
        f"({len(skipped)} skipped): {checkpoint_path}"
    )
    if skipped:
        preview = ", ".join(skipped[:8])
        suffix = "..." if len(skipped) > 8 else ""
        print(f"Skipped pretrain-only or incompatible tensors: {preview}{suffix}")
    return result


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
        "compile":         getattr(args, "compile", False),
        "data_parallel":   args.data_parallel,
        "grad_accum":      args.grad_accum_steps,
        "num_workers":     args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "pin_memory":      device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "non_blocking_transfer": device.type == "cuda",
        "seed":            args.seed,
        "train_embeddings_dir": args.train_embeddings_dir,
        "secondary_train_embeddings_dir": args.secondary_train_embeddings_dir,
        "token_train_embeddings_dir": args.token_train_embeddings_dir,
        "secondary_token_train_embeddings_dir": args.secondary_token_train_embeddings_dir,
        "tessera_presence_ch": args.tessera_presence_ch,
        "tessera_hidden_ch":   args.tessera_hidden_ch,
        "tessera_hidden_depth": args.tessera_hidden_depth,
        "height_specialist_depth": args.height_specialist_depth,
        "height_gate_source": args.height_gate_source,
        "height_hidden_ch": args.height_hidden_ch,
        "height_trunk_depth": args.height_trunk_depth,
        "height_independent_branches": args.height_independent_branches,
        "height_head_kind": args.height_head_kind,
        "height_n_bins": args.height_n_bins,
        "height_bin_max_m": args.height_bin_max_m,
        "height_bin_aux_weight": args.height_bin_aux_weight,
        "height_bin_sigma_bins": args.height_bin_sigma_bins,
        "building_smooth_weight": args.building_smooth_weight,
        "building_smooth_erode_px": args.building_smooth_erode_px,
        "building_smooth_thr": args.building_smooth_thr,
        "task": args.task,
        "lds_sampler": args.lds_sampler,
        "lds_bins":    args.lds_bins,
        "lds_sigma":   args.lds_sigma,
        "lds_cap":     args.lds_cap,
        "lds_h_max":   args.lds_h_max,
        "lds_score":   args.lds_score,
        "init_from_pretrain": args.init_from_pretrain,
        "init_pretrain_strict": args.init_pretrain_strict,
        "lightunet_base_ch":   args.lightunet_base_ch,
        "lightunet_norm_kind": args.lightunet_norm_kind,
        "gate_mode":           args.gate_mode,
        "gate_untied":         args.gate_untied,
        "gate_init_bias":      args.gate_init_bias,
        "modality_dropout":    args.modality_dropout,
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


def run_epoch(model, loader, criterion, optimizer, scaler, device, *, train,
              grad_accum_steps=1, use_amp=False, desc=""):
    """Train or eval one epoch. Returns (avg_loss, component_avgs)."""
    model.train(train)
    running_loss = 0.0
    component_sums = {}
    samples_seen = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    if train:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if train else torch.no_grad()
    non_blocking = device.type == "cuda"
    with context:
        for step, (imgs, targets, masks) in enumerate(pbar, start=1):
            imgs = move_to_device(imgs, device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            masks = masks.to(device, non_blocking=non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = forward_for_training(model, imgs)
                loss, loss_components = criterion(outputs, targets, masks)
                step_loss = loss / grad_accum_steps if train else loss

            if not torch.isfinite(loss):
                print(f"\nWARN: non-finite loss at step {step}; skipping batch.")
                if train:
                    optimizer.zero_grad(set_to_none=True)
                del outputs, loss, loss_components, step_loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            if train:
                scaler.scale(step_loss).backward()
                should_step = step % grad_accum_steps == 0 or step == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            bs = batch_size_of(imgs)
            running_loss += loss.item() * bs
            for name, value in loss_components.items():
                component_sums[name] = component_sums.get(name, 0.0) + value.detach().item() * bs
            samples_seen += bs
            avg = running_loss / max(1, samples_seen)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

    avg_loss = running_loss / max(1, samples_seen)
    comp_avg = {
        name: value / max(1, samples_seen)
        for name, value in component_sums.items()
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

    use_amp = args.amp and device.type == "cuda"
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
        height_gate_source=args.height_gate_source,
        height_hidden_ch=args.height_hidden_ch,
        height_trunk_depth=args.height_trunk_depth,
        height_independent_branches=args.height_independent_branches,
        height_head_kind=args.height_head_kind,
        height_n_bins=args.height_n_bins,
        height_bin_max_m=args.height_bin_max_m,
        lightunet_norm_kind=args.lightunet_norm_kind,
        gate_mode=args.gate_mode,
        gate_untied=args.gate_untied,
        gate_init_bias=args.gate_init_bias,
        modality_dropout=args.modality_dropout,
    )
    if args.init_from_pretrain:
        load_pretrain_weights(
            model,
            args.init_from_pretrain,
            strict=args.init_pretrain_strict,
        )
    model = model.to(device)
    if getattr(args, "compile", False) and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        model = model.to(memory_format=torch.channels_last)
        model = torch.compile(model, mode="default")
        print("torch.compile enabled (mode='default')")
    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            print(
                "WARN: --data-parallel requested but fewer than 2 CUDA devices "
                "are visible; continuing on one device."
            )
        else:
            model = torch.nn.DataParallel(model)
            print(f"Using DataParallel on {torch.cuda.device_count()} CUDA devices.")
    print(f"Using model: {selected_model} (input channels={n_channels})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
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
        height_bin_aux_weight=args.height_bin_aux_weight,
        height_bin_sigma_bins=args.height_bin_sigma_bins,
        building_smooth_weight=args.building_smooth_weight,
        building_smooth_erode_px=args.building_smooth_erode_px,
        building_smooth_thr=args.building_smooth_thr,
        task=args.task,
    ).to(device)
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
        f"focal_alpha={args.focal_alpha}, "
        f"height_head_kind={args.height_head_kind}, "
        f"height_n_bins={args.height_n_bins}, "
        f"height_bin_max_m={args.height_bin_max_m}, "
        f"height_bin_aux_weight={args.height_bin_aux_weight}, "
        f"height_bin_sigma_bins={args.height_bin_sigma_bins}, "
        f"building_smooth_weight={args.building_smooth_weight}, "
        f"building_smooth_erode_px={args.building_smooth_erode_px}, "
        f"building_smooth_thr={args.building_smooth_thr}, "
        f"task={args.task}"
    )

    print(f"Starting training on {device}...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        tr_loss, tr_comp = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, grad_accum_steps=grad_accum_steps, use_amp=use_amp,
            desc=f"Epoch {epoch + 1}/{args.epochs} [train]",
        )
        val_loss, val_comp = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, use_amp=use_amp,
            desc=f"Epoch {epoch + 1}/{args.epochs} [val]",
        )
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
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
            torch.save(state_dict_for_save(model), best_model_path)
            print(f"   >> New best val loss {best_val_loss:.4f} — saved.")

        print(f"Epoch {epoch + 1}/{args.epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
        print(f"   >> Train raw: {format_components(tr_comp, RAW_COMPONENTS)}")
        print(f"   >> Train weighted: {format_components(tr_comp, WEIGHTED_COMPONENTS)}")
        print(f"   >> Val raw:   {format_components(val_comp, RAW_COMPONENTS)}")
        print(f"   >> Val weighted: {format_components(val_comp, WEIGHTED_COMPONENTS)}")

    print("--- 3. Saving ---")
    torch.save(state_dict_for_save(model), last_model_path)
    plot_loss_curve(train_losses, val_losses, loss_curve_path, args.experiment_name)


if __name__ == "__main__":
    main()
