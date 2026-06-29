"""
Train a single emb2heights backbone.

The script is deliberately monolithic: one process trains one model on one
embedding source and writes all artifacts under `runs/<experiment_name>/`.
Compose multiple training runs in a shell script or slurm array — there is no
multi-baseline driver.
"""
import os
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
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
    set_cls_hole_config,
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
    "ae_only", "ae_tessera", "ae_tessera_gated", "ae_tessera_crossattn",
    "xfusion_crosslevel", "xfusion_pp", "tessera_token_crosslevel_s2_decoder64_perpixel",
    "hier_gated", "hierarchical_gated_token_fusion",
    "ae_tessera_simple", "ae_tessera_simple_gated",
    "ae_tessera_simple_convnext", "ae_tessera_simple_aspp",
    "simple_concat_fusion", "simple_gated_fusion",
    "simple_concat_convnext", "simple_concat_aspp",
    "ae_tessera_mlp", "ae_tessera_mlp_fusion",
    "ae_tessera_moe", "ae_tessera_moe_fusion",
    "ae_tessera_multilevel", "tessera_iou_fusion_multilevel_gated",
    "ae_tessera_segformer", "tessera_iou_fusion_segformer_lite",
    "multibackbone_fusion", "multi_backbone_fusion",
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
    "vegetation_smooth",
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
    "weighted_vegetation_smooth",
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
    p.add_argument("--gate-mode",
                   choices=["simple", "rich", "concat_mlp", "gated_mlp_residual",
                            "addmul", "film"],
                   default="simple",
                   help="Fusion mode for tessera_iou_fusion_gated. simple/rich = "
                        "tied sigmoid gate g·A+(1-g)·B (1x1 conv vs MLP gate). "
                        "concat_mlp = full nonlinear MLP fusion (no gate). "
                        "gated_mlp_residual = g·MLP(concat)+(1-g)·A. "
                        "addmul = g_add·A+(1-g_add)·B + g_mul·(A⊙B) with multiplicative term. "
                        "film = γ·A + β where γ,β are learned from B.")
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
    p.add_argument("--geom-aug", action="store_true",
                   help="Apply random D4 (h-flip/v-flip/transpose) augmentation to "
                        "training patches. Only wired into PixelEmbeddingDataset and "
                        "MultiPixelEmbeddingDataset (AE-only / AE+Tessera pixel paths).")
    p.add_argument("--cutmix-prob", type=float, default=0.0,
                   help="Probability of CutMix per training sample. Pastes a random box "
                        "from another training tile into the current one (both image and "
                        "label). Targets KE-density distribution shift by creating "
                        "synthetic high-density composites.")
    p.add_argument("--cutmix-min-frac", type=float, default=0.15)
    p.add_argument("--cutmix-max-frac", type=float, default=0.50)
    p.add_argument("--cutmix-density-aware", action="store_true",
                   help="When pasting via CutMix, prefer source tiles with higher "
                        "building density (creates KE-like dense composites).")
    p.add_argument("--mixup-prob", type=float, default=0.0,
                   help="Mixup probability per training sample. Linear blend of two "
                        "tiles (both embedding and label) with lam ~ Beta(alpha,alpha). "
                        "No discontinuities, unlike CutMix.")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Beta distribution alpha for mixup. 0.2 = strong mixing, 1.0 = mild.")
    p.add_argument("--bld-copypaste-prob", type=float, default=0.0,
                   help="Building copy-paste augmentation. Picks a donor tile with "
                        "buildings, finds the BBox of building pixels, pastes that "
                        "region (with corresponding embedding features) into the "
                        "current tile at a random position. Targets iou_bld.")
    p.add_argument("--bld-copypaste-max-size-frac", type=float, default=0.25,
                   help="Maximum size of the pasted building region as a fraction of "
                        "patch dimension. 0.25 = up to 25%% of patch h/w.")
    p.add_argument("--use-se", action="store_true",
                   help="Enable Squeeze-Excitation channel attention in LightUNet "
                        "DoubleConv blocks. Lightweight, well-proven on segmentation.")
    p.add_argument("--use-coord-attn", action="store_true",
                   help="Enable Coordinate Attention (Hou 2021) in LightUNet DoubleConv "
                        "blocks. Decomposes 2D pool into H+W pools so channel attention "
                        "carries positional info — relevant when regions matter (e.g. KE).")
    p.add_argument("--use-bottleneck-attn", action="store_true",
                   help="Insert a Transformer-style self-attention block at the UNet "
                        "bottleneck (32x32 at 256 input). Adds global context the conv "
                        "receptive field can't reach in 3 downsamples.")
    p.add_argument("--use-mixstyle", action="store_true",
                   help="MixStyle (Zhou 2021): mix per-sample feature statistics across "
                        "the batch during training to make the model robust to style/"
                        "domain shifts. Direct attack on the OOF→LB style gap.")
    p.add_argument("--use-attn-gates", action="store_true",
                   help="Attention U-Net skip gating (Oktay 2018): learn a sigmoid mask "
                        "on each skip connection conditioned on the decoder gating signal. "
                        "Filters encoder skip features so only relevant pixels flow into "
                        "the decoder, suppressing noise from KE-like dense regions.")
    p.add_argument("--use-aspp", action="store_true",
                   help="DeepLab-style Atrous Spatial Pyramid Pooling at the UNet "
                        "bottleneck (c4, 32x32 at 256 input). Parallel dilated convs at "
                        "rates {1,6,12,18} + global pool, GroupNorm. Expands receptive "
                        "field for multi-scale building context without downsampling.")
    p.add_argument("--bottleneck-attn-depth", type=int, default=1,
                   help="Number of Transformer blocks stacked at the UNet bottleneck "
                        "when --use-bottleneck-attn is on. depth=1 is the proven botattn "
                        "config; depth>1 deepens attention to test whether 1 block was a "
                        "bottleneck. Each block adds ~0.6M params at base_ch=48.")
    p.add_argument("--use-modern", action="store_true",
                   help="Modernize LightUNet conv blocks: add residual skip connections "
                        "(ResU-Net style) inside each DoubleConv + swap ReLU for GELU. "
                        "Targets gradient flow + smoother activations on deeper stacks.")
    p.add_argument("--use-xsource-fusion", action="store_true",
                   help="Replace single-token zero-init residual with CrossSourceHybridFiLMFusion "
                        "(ported from Dinghye xf085): N token sources self-attend, then each "
                        "contributes a zero-init FiLM γ/β + additive A + spatial gate σ(g) "
                        "residual to fused pixel features. Requires --token-train-embeddings-dir "
                        "pointing at concatenated 4-source tokens (3072 ch at 16x16).")
    p.add_argument("--token-source-ch", type=int, default=768,
                   help="Channels per token source (default 768 = TerraMind/THOR). "
                        "Total token channels must be divisible by this.")
    p.add_argument("--token-ctx-ch", type=int, default=96,
                   help="Per-source context channel width for xsource fusion (default 96).")
    p.add_argument("--xsource-attn-heads", type=int, default=4,
                   help="Multi-head attention heads in CrossSourceHybridFiLMFusion (default 4).")
    p.add_argument("--xsource-token-calibration", action="store_true",
                   help="Apply per-channel calibration to each token source before projection.")
    p.add_argument("--use-spatial-token-film", action="store_true",
                   help="Replace per-channel scalar token gate with xf095-style spatial-gated FiLM: "
                        "F_out = F + sigmoid(g(t)) * (gamma·F + beta + A(t)), where g is a 1x1 conv "
                        "to a single channel (spatial scalar gate per pixel). Single-source variant "
                        "of the xf095 win — targets RMSE_bH/vH which her ablation showed depend on "
                        "the spatial gate.")
    p.add_argument("--region-balanced-sampler", action="store_true",
                   help="WeightedRandomSampler with weights = 1 / count_of_tiles_in_region. "
                        "Equalizes per-region gradient contribution — each region (parsed from "
                        "filename suffix like _BE/_KE/_LL) gets equal effective batch share. "
                        "Targets distribution-shift between train and test region mix.")
    p.add_argument("--argmax-presence-target", action="store_true",
                   help="Use argmax across class fractions for the presence supervision "
                        "target (each multi-class pixel is positive ONLY for its dominant "
                        "class). Fixes the train/eval mismatch where fraction>0 trains the "
                        "model to predict every class at multi-class pixels.")
    p.add_argument("--argmax-bce-only", action="store_true",
                   help="When set with --argmax-presence-target, applies the argmax label "
                        "ONLY to the BCE presence loss; Tversky + height masks keep the "
                        "fraction>0 convention. Matches the narrower description of Ye "
                        "Dingqi's fix and avoids over-restricting height supervision.")
    p.add_argument("--argmax-include-bg", action="store_true",
                   help="With --argmax-presence-target: include BACKGROUND (1-sum(frac)) in "
                        "the argmax, so a pixel is positive for class c only if c beats the "
                        "other classes AND background. Fixes the labmate's noted bug (a "
                        "10%%-building/85%%-background pixel is now BACKGROUND, not building) "
                        "— directly kills the measured building↔background FP flooding.")
    p.add_argument("--building-presence-thr", type=float, default=0.0,
                   help="Per-class building presence threshold: building is a positive "
                        "presence/BCE target only where its fraction > thr (veg/water stay "
                        "frac>0). Default 0.0 == frac>0 (no-op). Gentle, tunable alternative "
                        "to --argmax-include-bg for cleaning building's 71%%-noisy frac>0 "
                        "label (label-noise audit 2026-06-24). Ignored if argmax targets on.")
    p.add_argument("--presence-coverage-thr", type=float, default=0.0,
                   help="Coverage threshold for the presence TARGET of ALL classes, aligned "
                        "to the official metric's GT binarization (coverage > 0.10, confirmed "
                        "2026-06-28). Default 0.0 = legacy frac>0 (misaligned). Set 0.10 to "
                        "train on the same positive set the leaderboard scores. Building may "
                        "override via --building-presence-thr.")
    p.add_argument("--cls-hole-mode", default="off", choices=["off", "exclude", "impute"],
                   help="Handle organizer CLASSIFICATION-mask blocks (landcover redacted "
                        "where a tall structure exists; symmetric to ndsm_hole). 'exclude' "
                        "drops cls-holes (no-landcover & height>thr) from the classification "
                        "loss; 'impute' sets a height-derived FAKE building label there. "
                        "Height stays supervised either way. 'off' = legacy.")
    p.add_argument("--cls-hole-h-thr", type=float, default=2.0,
                   help="Min height (m) for a cls-hole (no-landcover but a real structure).")
    p.add_argument("--water-argmax-bg", action="store_true",
                   help="Per-class WATER label: water presence/BCE target is positive "
                        "only where water beats {building, veg, background=1-sum(frac)} "
                        "(argmax-include-bg, water channel only). Building+veg stay frac>0 "
                        "(or building uses --building-presence-thr). Tests the labmate's "
                        "'argmax helps water' finding per-class (water 19%% frac>0 noise) "
                        "WITHOUT the global argmax-bg that guts building. Ignored if "
                        "--argmax-presence-target is on.")
    p.add_argument("--boundary-weight", type=float, default=0.0,
                   help="Enable boundary-aware loss. Upweights BCE on the building "
                        "channel near GT boundaries. >0 enables, e.g. 1.0 to turn on.")
    p.add_argument("--boundary-sigma-px", type=float, default=2.0,
                   help="Width of the boundary band in pixels (dilation iters).")
    p.add_argument("--boundary-amp", type=float, default=4.0,
                   help="Boundary upweight multiplier: weight = 1 + amp * boundary_band.")
    p.add_argument("--lovasz-weight", type=float, default=0.0,
                   help="Mix Lovász-Hinge loss on building logits with weight "
                        "lovasz_weight (added to total loss). Directly optimizes "
                        "building IoU. Recommended 0.3-0.7. 0 disables.")
    p.add_argument("--boundary-weight-vegwater", type=float, default=0.0,
                   help="Like --boundary-weight but upweights BCE on the VEG and "
                        "WATER channels near their GT boundaries. Targets the "
                        "veg/water boundary-recall leak. Reuses --boundary-sigma-px "
                        "and --boundary-amp. 0 disables.")
    p.add_argument("--lovasz-vegwater-weight", type=float, default=0.0,
                   help="Mix Lovász-Hinge (IoU surrogate) on veg+water logits with "
                        "this weight. Boundary-sensitive. 0 disables.")
    p.add_argument("--detail-bypass", action="store_true",
                   help="Add a full-res zero-init detail branch to the LightUNet "
                        "(bypasses the encoder/bottleneck) to retain fine spatial "
                        "detail. Starts as identity (== baseline).")
    p.add_argument("--encoder-arch", default="unet",
                   choices=["unet", "shallow", "dilated", "hrnet",
                            "unetpp", "unet_wave", "unetpp_wave"],
                   help="LightUNet encoder topology. 'unet' (default) = 3x "
                        "downsampling (unchanged baseline). 'shallow' = 2 "
                        "downsample stages (keeps more small-object detail). "
                        "'dilated' = NO downsampling, atrous convs grow the "
                        "receptive field at full res (directly preserves 1-4px "
                        "buildings the pooling erases). 'hrnet' = a full-res "
                        "stream maintained throughout with cross-res fusion. "
                        "Evidence: D1/D2 diagnosis — building loss is small "
                        "objects erased by 8x downsampling.")
    p.add_argument("--sharp-upsample", action="store_true",
                   help="Use PixelShuffle sub-pixel upsampling in the decoder "
                        "instead of bilinear, for sharper class boundaries.")
    p.add_argument("--scene-film", action="store_true",
                   help="Scene-conditioning FiLM: derive a global scene/region "
                        "descriptor from the LightUNet bottleneck and FiLM-modulate "
                        "the decoder output. Zero-init (starts == baseline).")
    p.add_argument("--distill-teacher-dir", type=str, default=None,
                   help="Directory of teacher predictions ({core_id}.npy, shape "
                        "(4,H,W): ch0-2 presence probs, ch3 height in METERS). "
                        "When set, enables DeiT-style knowledge distillation; "
                        "the dataset concatenates teacher channels onto the "
                        "target (-> 8-ch) and run_epoch adds a KD loss. Default "
                        "None = OFF (behavior byte-identical to before).")
    p.add_argument("--distill-weight", type=float, default=0.0,
                   help="Overall weight on the distillation loss (class BCE + "
                        "height SmoothL1 to the teacher). 0 disables KD even if "
                        "--distill-teacher-dir is set.")
    p.add_argument("--distill-height-weight", type=float, default=1.0,
                   help="Relative weight of the height KD term vs the class KD "
                        "term inside the distillation loss.")
    p.add_argument("--disable-head-film", action="store_true",
                   help="Disable FiLM-on-fractions inside the head. Labmate "
                        "found removing this improves height regression by "
                        "reducing gradient interference. Worth testing alongside "
                        "any new arch.")
    p.add_argument("--bidirectional-ctask", action="store_true",
                   help="Enable bidirectional cross-task attention in ae_tessera_gated: "
                        "height trunk features gate the presence head input via a "
                        "zero-initialized 1x1 conv (identity at init, learned gate). "
                        "No effect on ae_tessera_crossattn or ae_only.")
    p.add_argument("--dual-presence", action="store_true",
                   help="Add a parallel auxiliary presence branch (T-SwinUNet style). "
                        "It runs on the bare shared trunk features (no FiLM, no bidir, "
                        "no Tessera residual) and is supervised by the same BCE+Tversky. "
                        "Pair with --dual-presence-consistency-weight to enable an "
                        "IoU consistency loss between main and aux presence.")
    p.add_argument("--ae-only-deep-sup-weight", type=float, default=0.0,
                   help="Weight on CMGFNet-style deep supervision: a parallel "
                        "lightweight head on pre-fusion AE features predicts 4 "
                        "channels (presence + height) and is supervised with the "
                        "same BCE+Tversky+height losses. 0.0 disables (no aux head "
                        "is created in the model).")
    p.add_argument("--dual-presence-consistency-weight", type=float, default=0.0,
                   help="Weight for the soft-IoU consistency loss between main and "
                        "aux presence outputs. 0.0 disables the consistency term "
                        "(but the aux branch is still trained if --dual-presence is on).")
    p.add_argument("--height-blend-mode", default="presence_gated",
                   choices=["presence_gated", "max"],
                   help="How to blend per-class height specialists into the submitted "
                        "height channel. 'presence_gated' (legacy) = presence-weighted "
                        "convex blend with base_height fallback. 'max' = "
                        "max(building, vegetation, base) per pixel; decouples height "
                        "from presence calibration so a confident specialist routes "
                        "the pixel even if the presence head is uncertain.")
    p.add_argument("--crossattn-n-heads", type=int, default=4,
                   help="Number of attention heads for ae_tessera_crossattn bottleneck "
                        "cross-attention. embed_dim=base_ch*8 must be divisible by this.")
    p.add_argument("--train-targets-dir",    default=DEFAULT_TRAIN_TAR)
    p.add_argument("--experiment-name",      default=DEFAULTS["experiment_name"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--patch-size",     type=int,   default=DEFAULTS["patch_size"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay",   type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--lr-schedule", choices=["plateau", "cosine"], default="plateau",
                   help="LR schedule. 'plateau' = ReduceLROnPlateau (conv default). "
                        "'cosine' = linear warmup then cosine decay to 0 over --epochs "
                        "(proper schedule for training transformers from scratch).")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup epochs (only used with --lr-schedule cosine).")
    p.add_argument("--no-wd-on-norm", action="store_true",
                   help="Exclude LayerNorm/BatchNorm weights and biases from weight decay "
                        "(standard transformer-from-scratch practice).")
    p.add_argument("--vit-drop-rate", type=float, default=0.0,
                   help="Dropout rate inside SegFormer attention + Mix-FFN (AugReg "
                        "regularization for from-scratch ViT). Training-only, param-free.")
    p.add_argument("--vit-drop-path-rate", type=float, default=0.0,
                   help="Stochastic-depth (drop-path) max rate, linearly scheduled across "
                        "SegFormer blocks. Training-only, param-free.")
    p.add_argument("--pretrained-backbone-path", type=str, default=None,
                   help="Path to a pretrained remote-sensing backbone body (.pth) for "
                        "model-type multibackbone_fusion. Loaded with strict=False into a "
                        "timm ResNet50 body; the input stem is re-init for 192 channels. "
                        "Omit for the random-init control.")
    p.add_argument("--backbone-input-proj-ch", type=int, default=None,
                   help="If set (e.g. 3 or 13), build a 192->proj_ch input adapter so the "
                        "pretrained ResNet50 stem (conv1) is KEPT/loaded instead of re-init. "
                        "Default None = current behavior (in_chans=192, stem re-init).")
    p.add_argument("--backbone-input-norm", type=str, default=None,
                   choices=[None, "imagenet", "instance"],
                   help="Per-channel normalization of the adapter output before the backbone. "
                        "'imagenet' (proj_ch==3 only) or 'instance' (any proj_ch).")
    p.add_argument("--backbone-pretrained-source", type=str, default=None,
                   help="Pretrained source for the backbone: 'imagenet' (timm pretrained=True), "
                        "a path to a full state_dict .pth, or None (random). Takes precedence "
                        "over --pretrained-backbone-path when set.")
    p.add_argument("--freeze-backbone-stages", type=int, default=0,
                   help="Freeze early pretrained backbone groups to preserve spatial knowledge: "
                        "0=none, 1=stem, 2=stem+layer1, 3=stem+layer1+layer2, etc.")
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
                   choices=["l1", "huber", "mse", "berhu"],
                   help="Regression loss used for height_boost and aux class-height "
                        "supervision. Default l1 matches legacy behavior. berhu = "
                        "reverse Huber: L1 for small errors, quadratic for large ones "
                        "(c = 0.2 * max per batch). Penalizes building-edge residuals "
                        "more than L1 without the instability of pure MSE.")
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
    p.add_argument("--height-dropout", type=float, default=0.0,
                   help="Dropout2d on the height-trunk input only (presence path "
                        "untouched). Regularizes the height path so it can't memorize "
                        "local-region height stats → better test/LB height generalization "
                        "(heights are the biggest local→public leak). 0.0 = off. Try 0.1.")
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
    p.add_argument("--vegetation-smooth-weight", type=float, default=0.0,
                   help="Weight for height total-variation loss inside eroded GT "
                        "vegetation interiors. 0.0 disables it. Lighter than "
                        "building smooth is usually appropriate because canopy "
                        "height varies smoothly rather than being uniform.")
    p.add_argument("--vegetation-smooth-erode-px", type=int, default=1,
                   help="Binary erosion radius in pixels before applying "
                        "vegetation interior smoothness.")
    p.add_argument("--vegetation-smooth-thr", type=float, default=0.0,
                   help="GT vegetation fraction threshold used to define "
                        "vegetation pixels for interior smoothness.")
    p.add_argument("--building-presence-tversky-weight", type=float, default=1.0,
                   help="Per-channel weight multiplier applied to the building "
                        "class in the presence Tversky loss. 1.0 = symmetric "
                        "(default); >1 emphasizes building recall — target "
                        "iou_bld when buildings are the LB bottleneck.")
    p.add_argument("--building-presence-bce-weight", type=float, default=1.0,
                   help="Per-channel weight multiplier applied to the building "
                        "class in the presence BCE loss. 1.0 = symmetric; "
                        ">1 emphasizes building presence prediction.")
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
    p.add_argument("--use-shape-queries", action="store_true",
                   help="Add the zero-init QueryShapeRefiner (MaskFormer-lite semantic "
                        "mask-classification) as a residual on the presence logits — "
                        "injects object-shape priors; starts as identity (== base model).")
    p.add_argument("--shape-n-queries", type=int, default=32,
                   help="Number of learned object queries for --use-shape-queries.")
    p.add_argument("--shape-depth", type=int, default=2,
                   help="Transformer decoder layers for the shape-query refiner.")
    p.add_argument("--swa", action="store_true",
                   help="Stochastic Weight Averaging: from --swa-start-frac of epochs, "
                        "average weights at a constant --swa-lr; recompute BN at the end "
                        "and save the averaged model as model_best.pth. Flat-minima / "
                        "domain-generalization regularizer. Auto-disables torch.compile.")
    p.add_argument("--swa-start-frac", type=float, default=0.7,
                   help="Fraction of total epochs after which SWA averaging begins.")
    p.add_argument("--swa-lr", type=float, default=1e-4,
                   help="Constant LR held during the SWA averaging phase.")
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
    p.add_argument("--presence-grad-scale", type=float, default=1.0,
                   help="Two-stage 'purify' knob. 1.0 (default) = fully coupled "
                        "seg+height training (Stage 1, no behavior change). 0.0 = "
                        "seg/presence losses no longer update the shared backbone, "
                        "so only the height path re-tunes it (Stage 2). Combine "
                        "with --select-on height and --init-from-pretrain <stage1>.")
    p.add_argument("--height-grad-scale", type=float, default=1.0,
                   help="Mirror of --presence-grad-scale for the Stage-3 'seg-purify' "
                        "(labmate's +5th-place lever). 0.0 = height/fraction losses no "
                        "longer update the backbone, so only segmentation re-tunes it. "
                        "Combine with --select-on loss + --init-from-pretrain <stage2>.")
    p.add_argument("--select-on", choices=("loss", "height"), default="loss",
                   help="Which validation metric selects model_best.pth. 'loss' "
                        "(default) = total composite val loss. 'height' = weighted "
                        "height terms only (height_boost + aux_height + bin_ce); "
                        "use for the Stage-2 height-purify checkpoint.")
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                   help="Warm-start gentle integration: freeze the transferred "
                        "backbone modules (alpha_unet/tessera_feature_stem/"
                        "gate_conv/xsource_fusion) for the first N epochs so the "
                        "fresh heads adapt to the pretrained features before the "
                        "backbone moves. Frozen modules' BatchNorm is set to eval "
                        "(AMP-safe — avoids fp16 corrupting frozen BN). 0 = off.")
    p.add_argument("--unfreeze-warmup-epochs", type=int, default=3,
                   help="After the backbone unfreezes, linearly ramp the LR from "
                        "0 to the scheduled value over this many epochs to avoid an "
                        "unfreeze shock. Used with --freeze-backbone-epochs.")
    p.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16",
                   help="Autocast precision. 'fp16' (default, legacy, needs GradScaler "
                        "and can corrupt BatchNorm). 'bf16' = bfloat16: same speed on "
                        "A100, numerically stable (no GradScaler, no BN corruption, no "
                        "NaN-skip batches). Recommended for these BN models.")
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

    # Configure classification-mask handling BEFORE datasets/DataLoader workers fork.
    set_cls_hole_config(getattr(args, "cls_hole_mode", "off"),
                        getattr(args, "cls_hole_h_thr", 2.0))

    teacher_dir = getattr(args, "distill_teacher_dir", None)
    # Only datasets known to support KD accept the distill_teacher_dir kwarg.
    distill_kwargs = (
        {"distill_teacher_dir": teacher_dir}
        if teacher_dir and DatasetCls.__name__ in {
            "PixelEmbeddingDataset",
            "MultiPixelEmbeddingDataset",
            "PixelTokenEmbeddingDataset",
        }
        else {}
    )

    if DatasetCls.__name__ == "LatentTokenDataset":
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size, scale_factor=16, is_train=True)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size, scale_factor=16, is_train=False)
    elif DatasetCls.__name__ in {"PixelTokenEmbeddingDataset", "PixelMultiTokenEmbeddingDataset", "MultiLatentTokenDataset"}:
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size, scale_factor=16, is_train=True, **distill_kwargs)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size, scale_factor=16, is_train=False)
    else:
        ds_kwargs = dict(
            patch_size=args.patch_size,
            is_train=True,
            geom_aug=getattr(args, "geom_aug", False),
        )
        ds_kwargs.update(distill_kwargs)
        # Only the multi-pixel dataset supports CutMix
        if DatasetCls.__name__ == "MultiPixelEmbeddingDataset":
            ds_kwargs.update(
                cutmix_prob=getattr(args, "cutmix_prob", 0.0),
                cutmix_min_frac=getattr(args, "cutmix_min_frac", 0.15),
                cutmix_max_frac=getattr(args, "cutmix_max_frac", 0.50),
                cutmix_density_aware=getattr(args, "cutmix_density_aware", False),
                mixup_prob=getattr(args, "mixup_prob", 0.0),
                mixup_alpha=getattr(args, "mixup_alpha", 0.2),
                bld_copypaste_prob=getattr(args, "bld_copypaste_prob", 0.0),
                bld_copypaste_max_size_frac=getattr(args, "bld_copypaste_max_size_frac", 0.25),
            )
        train_ds = DatasetCls(train_pairs, **ds_kwargs)
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

    def _strip(k):
        for prefix in ("module.", "_orig_mod."):
            while k.startswith(prefix):
                k = k[len(prefix):]
        return k

    state = {_strip(k): v for k, v in state.items()}

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
        "height_dropout": getattr(args, "height_dropout", 0.0),
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
        "vegetation_smooth_weight": args.vegetation_smooth_weight,
        "vegetation_smooth_erode_px": args.vegetation_smooth_erode_px,
        "vegetation_smooth_thr": args.vegetation_smooth_thr,
        "building_presence_tversky_weight": args.building_presence_tversky_weight,
        "building_presence_bce_weight": args.building_presence_bce_weight,
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
        "bidirectional_ctask": args.bidirectional_ctask,
        "height_blend_mode":   args.height_blend_mode,
        "dual_presence":       args.dual_presence,
        "dual_presence_consistency_weight": args.dual_presence_consistency_weight,
        "ae_only_deep_sup_weight": args.ae_only_deep_sup_weight,
        "height_loss_kind":    args.height_loss_kind,
        "huber_delta":         args.huber_delta,
        "build_height_boost":  args.build_height_boost,
        "veg_height_boost":    args.veg_height_boost,
        "aux_veg_weight":      args.aux_veg_weight,
        "iou_loss_kind":       args.iou_loss_kind,
        "focal_gamma":         args.focal_gamma,
        "focal_alpha":         args.focal_alpha,
        "use_se":              getattr(args, "use_se", False),
        "use_coord_attn":      getattr(args, "use_coord_attn", False),
        "use_bottleneck_attn": getattr(args, "use_bottleneck_attn", False),
        "use_mixstyle":        getattr(args, "use_mixstyle", False),
        "use_attn_gates":      getattr(args, "use_attn_gates", False),
        "use_aspp":            getattr(args, "use_aspp", False),
        "bottleneck_attn_depth": getattr(args, "bottleneck_attn_depth", 1),
        "use_modern":          getattr(args, "use_modern", False),
        "use_xsource_fusion":  getattr(args, "use_xsource_fusion", False),
        "token_source_ch":     getattr(args, "token_source_ch", 768),
        "token_ctx_ch":        getattr(args, "token_ctx_ch", 96),
        "xsource_attn_heads":  getattr(args, "xsource_attn_heads", 4),
        "xsource_token_calibration": getattr(args, "xsource_token_calibration", False),
        "use_spatial_token_film": getattr(args, "use_spatial_token_film", False),
        # MultiBackboneFusion architecture params — required to rebuild the
        # identical model at predict time (proj_ch drives the stem in_chans).
        "pretrained_backbone_path": getattr(args, "pretrained_backbone_path", None),
        "backbone_input_proj_ch": getattr(args, "backbone_input_proj_ch", None),
        "backbone_input_norm": getattr(args, "backbone_input_norm", None),
        "backbone_pretrained_source": getattr(args, "backbone_pretrained_source", None),
        "freeze_backbone_stages": getattr(args, "freeze_backbone_stages", 0),
        "use_shape_queries": getattr(args, "use_shape_queries", False),
        "shape_n_queries": getattr(args, "shape_n_queries", 32),
        "shape_depth": getattr(args, "shape_depth", 2),
        "distill_teacher_dir": getattr(args, "distill_teacher_dir", None),
        "distill_weight":      getattr(args, "distill_weight", 0.0),
        "distill_height_weight": getattr(args, "distill_height_weight", 1.0),
        "argmax_presence_target": getattr(args, "argmax_presence_target", False),
        "argmax_include_bg": getattr(args, "argmax_include_bg", False),
        "argmax_bce_only":     getattr(args, "argmax_bce_only", False),
        "building_presence_thr": getattr(args, "building_presence_thr", 0.0),
        "presence_coverage_thr": getattr(args, "presence_coverage_thr", 0.0),
        "water_argmax_bg":       getattr(args, "water_argmax_bg", False),
        "cls_hole_mode":         getattr(args, "cls_hole_mode", "off"),
        "cls_hole_h_thr":        getattr(args, "cls_hole_h_thr", 2.0),
        "disable_head_film":   getattr(args, "disable_head_film", False),
        "lovasz_weight":       getattr(args, "lovasz_weight", 0.0),
        "boundary_weight":     getattr(args, "boundary_weight", 0.0),
        "boundary_sigma_px":   getattr(args, "boundary_sigma_px", 2.0),
        "boundary_amp":        getattr(args, "boundary_amp", 4.0),
        "boundary_weight_vegwater": getattr(args, "boundary_weight_vegwater", 0.0),
        "lovasz_vegwater_weight": getattr(args, "lovasz_vegwater_weight", 0.0),
        "detail_bypass":       getattr(args, "detail_bypass", False),
        "sharp_upsample":      getattr(args, "sharp_upsample", False),
        "scene_film":          getattr(args, "scene_film", False),
        "encoder_arch":        getattr(args, "encoder_arch", "unet"),
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


def _distillation_loss(student_out, teacher, masks, height_weight):
    """DeiT-style KD: class BCE + weighted height SmoothL1 to a teacher.

    Args:
        student_out: (B,4,H,W) student output; ch0-2 presence PROBABILITIES,
                     ch3 height on the model's (normalized) output scale.
        teacher:     (B,4,H,W) teacher; ch0-2 presence probabilities, ch3 height
                     already converted to the same normalized scale by the dataset.
        masks:       (B,2,H,W) validity; ch0 = global, ch1 = height validity.
        height_weight: relative weight of the height term vs the class term.
    """
    global_mask = masks[:, 0:1, :, :]
    height_mask = masks[:, 1:2, :, :]

    # BCE computed manually in fp32: F.binary_cross_entropy is unsafe under AMP
    # autocast (it takes probabilities, not logits) and raised RuntimeError.
    student_probs = student_out[:, :3, :, :].clamp(1e-6, 1.0 - 1e-6).float()
    teacher_probs = teacher[:, :3, :, :].clamp(0.0, 1.0).float()
    cls_bce = -(teacher_probs * torch.log(student_probs)
                + (1.0 - teacher_probs) * torch.log(1.0 - student_probs))
    cls_mask = global_mask.expand(-1, 3, -1, -1)
    cls_loss = (cls_bce * cls_mask).sum() / (cls_mask.sum() + 1e-6)

    h_err = torch.nn.functional.smooth_l1_loss(
        student_out[:, 3:4, :, :], teacher[:, 3:4, :, :], reduction="none"
    )
    h_loss = (h_err * height_mask).sum() / (height_mask.sum() + 1e-6)

    return cls_loss + height_weight * h_loss


def run_epoch(model, loader, criterion, optimizer, scaler, device, *, train,
              grad_accum_steps=1, use_amp=False, desc="",
              distill_weight=0.0, distill_height_weight=1.0,
              amp_dtype=torch.float16):
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

            # Knowledge distillation: when enabled, the dataset has concatenated
            # 4 teacher channels onto the target (-> 8 channels). Split them off
            # so the criterion sees a normal 4-channel target, and add a KD term.
            do_distill = (
                train
                and distill_weight > 0
                and targets.dim() == 4
                and targets.shape[1] >= 8
            )
            if targets.dim() == 4 and targets.shape[1] >= 8:
                teacher_t = targets[:, 4:8, :, :]
                targets = targets[:, :4, :, :]
            else:
                teacher_t = None

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = forward_for_training(model, imgs)
                loss, loss_components = criterion(outputs, targets, masks)
                if do_distill and teacher_t is not None:
                    student_out = outputs["out"] if isinstance(outputs, dict) else outputs
                    distill_loss = _distillation_loss(
                        student_out, teacher_t, masks, distill_height_weight
                    )
                    loss = loss + distill_weight * distill_loss
                    loss_components = dict(loss_components)
                    loss_components["distill"] = distill_weight * distill_loss
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
        height_dropout=getattr(args, "height_dropout", 0.0),
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
        bidirectional_ctask=args.bidirectional_ctask,
        crossattn_n_heads=args.crossattn_n_heads,
        height_blend_mode=args.height_blend_mode,
        dual_presence=args.dual_presence,
        ae_only_supervision=(args.ae_only_deep_sup_weight > 0.0),
        use_se=getattr(args, "use_se", False),
        use_coord_attn=getattr(args, "use_coord_attn", False),
        use_bottleneck_attn=getattr(args, "use_bottleneck_attn", False),
        use_mixstyle=getattr(args, "use_mixstyle", False),
        use_attn_gates=getattr(args, "use_attn_gates", False),
        use_aspp=getattr(args, "use_aspp", False),
        bottleneck_attn_depth=getattr(args, "bottleneck_attn_depth", 1),
        use_modern=getattr(args, "use_modern", False),
        detail_bypass=getattr(args, "detail_bypass", False),
        sharp_upsample=getattr(args, "sharp_upsample", False),
        scene_film=getattr(args, "scene_film", False),
        encoder_arch=getattr(args, "encoder_arch", "unet"),
        disable_head_film=getattr(args, "disable_head_film", False),
        use_xsource_fusion=getattr(args, "use_xsource_fusion", False),
        token_source_ch=getattr(args, "token_source_ch", 768),
        token_ctx_ch=getattr(args, "token_ctx_ch", 96),
        xsource_attn_heads=getattr(args, "xsource_attn_heads", 4),
        xsource_token_calibration=getattr(args, "xsource_token_calibration", False),
        use_spatial_token_film=getattr(args, "use_spatial_token_film", False),
        vit_drop_rate=getattr(args, "vit_drop_rate", 0.0),
        vit_drop_path_rate=getattr(args, "vit_drop_path_rate", 0.0),
        pretrained_backbone_path=getattr(args, "pretrained_backbone_path", None),
        backbone_input_proj_ch=getattr(args, "backbone_input_proj_ch", None),
        backbone_input_norm=getattr(args, "backbone_input_norm", None),
        backbone_pretrained_source=getattr(args, "backbone_pretrained_source", None),
        freeze_backbone_stages=getattr(args, "freeze_backbone_stages", 0),
        use_shape_queries=getattr(args, "use_shape_queries", False),
        shape_n_queries=getattr(args, "shape_n_queries", 32),
        shape_depth=getattr(args, "shape_depth", 2),
    )
    if args.init_from_pretrain:
        load_pretrain_weights(
            model,
            args.init_from_pretrain,
            strict=args.init_pretrain_strict,
        )
    # Two-stage purify: scale the gradient that seg/presence losses send to the
    # shared backbone (set on every head exposing the knob). 1.0 = no-op.
    pg_scale = getattr(args, "presence_grad_scale", 1.0)
    if pg_scale != 1.0:
        n_set = sum(
            (setattr(m, "presence_grad_scale", float(pg_scale)) or 1)
            for m in model.modules() if hasattr(m, "presence_grad_scale")
        )
        print(f"Two-stage purify: presence_grad_scale={pg_scale} set on {n_set} head(s).")
    hg_scale = getattr(args, "height_grad_scale", 1.0)
    if hg_scale != 1.0:
        n_set = sum(
            (setattr(m, "height_grad_scale", float(hg_scale)) or 1)
            for m in model.modules() if hasattr(m, "height_grad_scale")
        )
        print(f"Seg-purify: height_grad_scale={hg_scale} set on {n_set} head(s).")
    model = model.to(device)
    if getattr(args, "compile", False) and device.type == "cuda" and not getattr(args, "swa", False):
        torch.set_float32_matmul_precision("high")
        model = model.to(memory_format=torch.channels_last)
        model = torch.compile(model, mode="default")
        print("torch.compile enabled (mode='default')")
    elif getattr(args, "compile", False) and getattr(args, "swa", False):
        print("torch.compile DISABLED because --swa is set (AveragedModel deep-copy "
              "is incompatible with compiled modules).")
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

    if args.no_wd_on_norm:
        # 1-D params (norm weights) and biases get no weight decay; standard
        # transformer-from-scratch practice. Everything else decays as usual.
        decay_params, no_decay_params = [], []
        for p_name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or p_name.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.lr,
        )
        print(f"AdamW param groups: {len(decay_params)} decayed, "
              f"{len(no_decay_params)} no-decay (norms/biases).")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "fp16") == "bf16" else torch.float16
    # GradScaler is only needed/valid for fp16; bf16 has fp32 dynamic range.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if use_amp:
        print(f"AMP autocast dtype: {('bfloat16' if amp_dtype==torch.bfloat16 else 'float16')} "
              f"(GradScaler {'on' if amp_dtype==torch.float16 else 'off'})")

    use_plateau = args.lr_schedule == "plateau"
    if use_plateau:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    else:
        # Linear warmup over --warmup-epochs, then cosine decay to ~0 over the
        # remaining epochs. The plateau scheduler collapsed the ViT LR to ~1e-6
        # by epoch ~50, effectively freezing it; cosine keeps the LR productive.
        _warmup = max(0, args.warmup_epochs)
        _total = max(1, args.epochs)

        def _lr_lambda(ep):
            if _warmup > 0 and ep < _warmup:
                return float(ep + 1) / float(_warmup)
            progress = (ep - _warmup) / max(1, _total - _warmup)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(f"LR schedule: cosine, warmup={_warmup} epochs, total={_total}.")

    # --- Warm-start gentle integration: freeze the transferred backbone for the
    # first N epochs (BN→eval so AMP can't corrupt frozen stats), then unfreeze
    # with an LR warmup. The optimizer above already holds ALL params, so toggling
    # requires_grad gates updates without rebuilding it. No-op when N==0. ---
    _FREEZE_PREFIXES = ("alpha_unet", "tessera_feature_stem", "gate_conv", "xsource_fusion")
    freeze_epochs = int(getattr(args, "freeze_backbone_epochs", 0))
    unfreeze_warmup = max(1, int(getattr(args, "unfreeze_warmup_epochs", 3)))
    _bm = model.module if isinstance(model, torch.nn.DataParallel) else model

    def _set_backbone_frozen(frozen):
        n = 0
        for pname, p in _bm.named_parameters():
            if any(pname.startswith(pre) for pre in _FREEZE_PREFIXES):
                p.requires_grad = (not frozen)
                n += 1
        for mname, m in _bm.named_modules():
            if isinstance(m, nn.BatchNorm2d) and any(
                mname.startswith(pre) for pre in _FREEZE_PREFIXES
            ):
                m.eval() if frozen else m.train()
        return n

    if freeze_epochs > 0:
        _nfz = _set_backbone_frozen(True)
        print(f"Warm-start freeze: {_nfz} backbone params frozen (BN→eval) for "
              f"{freeze_epochs} epochs, then {unfreeze_warmup}-epoch LR warmup.")
    resolved_loss_preset = resolve_loss_preset(args)
    loss_lambdas = effective_loss_lambdas(args)
    criterion = ImprovedCompositeLoss(
        weight_mae=loss_lambdas[0],
        weight_height_boost=loss_lambdas[3],
        aux_weight=args.aux_weight,
        loss_preset=resolved_loss_preset,
        weight_presence_tversky=args.presence_tversky_weight,
        weight_fraction_mae=args.fraction_mae_weight,
        height_loss_kind=args.height_loss_kind,
        huber_delta=args.huber_delta,
        build_height_boost=args.build_height_boost,
        veg_height_boost=args.veg_height_boost,
        aux_veg_weight=args.aux_veg_weight,
        height_bin_aux_weight=args.height_bin_aux_weight,
        height_bin_sigma_bins=args.height_bin_sigma_bins,
        dual_presence_consistency_weight=args.dual_presence_consistency_weight,
        ae_only_deep_sup_weight=args.ae_only_deep_sup_weight,
        building_smooth_weight=args.building_smooth_weight,
        building_smooth_erode_px=args.building_smooth_erode_px,
        building_smooth_thr=args.building_smooth_thr,
        vegetation_smooth_weight=args.vegetation_smooth_weight,
        vegetation_smooth_erode_px=args.vegetation_smooth_erode_px,
        vegetation_smooth_thr=args.vegetation_smooth_thr,
        building_presence_tversky_weight=args.building_presence_tversky_weight,
        building_presence_bce_weight=args.building_presence_bce_weight,
        boundary_weight=getattr(args, "boundary_weight", 0.0),
        boundary_sigma_px=getattr(args, "boundary_sigma_px", 2.0),
        boundary_amp=getattr(args, "boundary_amp", 4.0),
        boundary_weight_vegwater=getattr(args, "boundary_weight_vegwater", 0.0),
        lovasz_weight=getattr(args, "lovasz_weight", 0.0),
        lovasz_vegwater_weight=getattr(args, "lovasz_vegwater_weight", 0.0),
        argmax_presence_target=getattr(args, "argmax_presence_target", False),
        argmax_bce_only=getattr(args, "argmax_bce_only", False),
        argmax_include_bg=getattr(args, "argmax_include_bg", False),
        building_presence_thr=getattr(args, "building_presence_thr", 0.0),
        presence_coverage_thr=getattr(args, "presence_coverage_thr", 0.0),
        water_argmax_bg=getattr(args, "water_argmax_bg", False),
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
        f"vegetation_smooth_weight={args.vegetation_smooth_weight}, "
        f"vegetation_smooth_erode_px={args.vegetation_smooth_erode_px}, "
        f"vegetation_smooth_thr={args.vegetation_smooth_thr}, "
        f"task={args.task}"
    )

    swa_model = None
    swa_start = args.epochs
    if getattr(args, "swa", False):
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_start = int(args.swa_start_frac * args.epochs)
        print(f"SWA enabled: averaging weights from epoch {swa_start + 1}/{args.epochs} "
              f"at constant lr={args.swa_lr}.")

    print(f"Starting training on {device}...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Warm-start: unfreeze the backbone once the freeze window ends, then
        # ramp LR over `unfreeze_warmup` epochs (multiplier applied on top of the
        # scheduler's LR; LambdaLR recomputes from base_lrs so this never compounds).
        if freeze_epochs > 0 and epoch == freeze_epochs:
            _set_backbone_frozen(False)
            print(f"   >> Unfroze backbone at epoch {epoch + 1} "
                  f"(LR warmup over {unfreeze_warmup} epochs).")
        if freeze_epochs > 0 and freeze_epochs <= epoch < freeze_epochs + unfreeze_warmup:
            _ramp = float(epoch - freeze_epochs + 1) / float(unfreeze_warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * _ramp
        tr_loss, tr_comp = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, grad_accum_steps=grad_accum_steps, use_amp=use_amp,
            desc=f"Epoch {epoch + 1}/{args.epochs} [train]",
            distill_weight=getattr(args, "distill_weight", 0.0),
            distill_height_weight=getattr(args, "distill_height_weight", 1.0),
            amp_dtype=amp_dtype,
        )
        val_loss, val_comp = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, use_amp=use_amp,
            desc=f"Epoch {epoch + 1}/{args.epochs} [val]",
            amp_dtype=amp_dtype,
        )
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        if swa_model is not None and epoch >= swa_start:
            for pg in optimizer.param_groups:
                pg["lr"] = args.swa_lr
            swa_model.update_parameters(model)
            print(f"   >> SWA: averaged epoch {epoch + 1} (n_averaged="
                  f"{int(swa_model.n_averaged.item())}).")
        elif use_plateau:
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

        if args.select_on == "height":
            select_metric = (
                float(val_comp.get("weighted_height_boost", 0.0))
                + float(val_comp.get("weighted_aux_height", 0.0))
                + float(val_comp.get("weighted_height_bin_ce", 0.0))
            )
        else:
            select_metric = val_loss
        if select_metric < best_val_loss:
            best_val_loss = select_metric
            torch.save(state_dict_for_save(model), best_model_path)
            _label = "height val loss" if args.select_on == "height" else "val loss"
            print(f"   >> New best {_label} {best_val_loss:.4f} — saved.")

        print(f"Epoch {epoch + 1}/{args.epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
        print(f"   >> Train raw: {format_components(tr_comp, RAW_COMPONENTS)}")
        print(f"   >> Train weighted: {format_components(tr_comp, WEIGHTED_COMPONENTS)}")
        print(f"   >> Val raw:   {format_components(val_comp, RAW_COMPONENTS)}")
        print(f"   >> Val weighted: {format_components(val_comp, WEIGHTED_COMPONENTS)}")

    print("--- 3. Saving ---")
    torch.save(state_dict_for_save(model), last_model_path)
    if swa_model is not None and int(swa_model.n_averaged.item()) > 0:
        from torch.optim.swa_utils import update_bn
        print(f"--- SWA: recomputing BN statistics over training data "
              f"(n_averaged={int(swa_model.n_averaged.item())}) ---")
        update_bn(train_loader, swa_model, device=device)
        torch.save(state_dict_for_save(swa_model.module), best_model_path)
        torch.save(state_dict_for_save(swa_model.module),
                   os.path.join(exp_dir, "model_swa.pth"))
        print("SWA weights saved as model_best.pth (and model_swa.pth).")
    plot_loss_curve(train_losses, val_losses, loss_curve_path, args.experiment_name)


if __name__ == "__main__":
    main()
