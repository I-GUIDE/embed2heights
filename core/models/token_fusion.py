import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ChannelCalibration


class CrossSourceHybridFiLMFusion(nn.Module):
    """xf085 SoTA fusion: cross-source self-attention + per-source FiLM + additive + spatial gate.

    Stage 1 (16x16 token scale): N token sources cross-attend to each other via
    one self-attention layer with learned modality embeddings + 2D positional
    encoding. Output projection is zero-initialised => refined ~ projected at init.

    Stage 2 (H x W pixel scale): each refined source contributes three zero-init
    residuals applied as
        delta_i = sigmoid(g_i) * (gamma_i * F_pixel + beta_i + A_i)
        F_out   = F_pixel + sum_i delta_i

    All three per-source pathways (FiLM gamma/beta, additive A_i, spatial gate
    sigma(g_i)) are always built; score-based attribution showed each off-toggle
    degraded the leaderboard score, so they are not configurable.
    """

    _TOKEN_INPUT_CLAMP = 50.0
    _CTX_CLAMP = 50.0
    _FILM_PARAM_CLAMP = 4.0
    _ADD_CLAMP = 4.0

    def __init__(self, pixel_ch, token_channels, token_source_ch=768,
                 ctx_ch=96, token_calibration=False,
                 attn_heads=4, attn_dropout=0.05,
                 token_calibration_source_indices=None):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"CrossSourceHybridFiLMFusion: token_channels={token_channels} must be "
                f"divisible by token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.n_sources = token_channels // token_source_ch
        self.ctx_ch = int(ctx_ch)
        self.attn_heads = int(attn_heads)

        # Selective calibration: when token_calibration_source_indices is given,
        # only those source indices get a learnable ChannelCalibration; others
        # bypass through nn.Identity (forward indexes unchanged). xf095 uses
        # indices=[0,1] (terramind only) -- THOR's raw ±17000 scale must NOT be
        # calibrated or iou_wat collapses.
        if token_calibration:
            if token_calibration_source_indices is None:
                calib_set = set(range(self.n_sources))
            else:
                calib_set = {int(i) for i in token_calibration_source_indices}
            self.token_calibs = nn.ModuleList([
                ChannelCalibration(token_source_ch) if i in calib_set
                else nn.Identity()
                for i in range(self.n_sources)
            ])
        else:
            self.token_calibs = None
        self.token_projs = nn.ModuleList([
            nn.Conv2d(token_source_ch, ctx_ch, 1, bias=False)
            for _ in range(self.n_sources)
        ])

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, ctx_ch),
            nn.GELU(),
            nn.Linear(ctx_ch, ctx_ch),
        )

        self.modality_embed = nn.Parameter(torch.zeros(self.n_sources, ctx_ch))
        nn.init.normal_(self.modality_embed, std=0.02)
        self.attn_norm = nn.LayerNorm(ctx_ch)
        self.cross_source_attn = nn.MultiheadAttention(
            ctx_ch, self.attn_heads, dropout=attn_dropout, batch_first=True,
        )
        nn.init.zeros_(self.cross_source_attn.out_proj.weight)
        nn.init.zeros_(self.cross_source_attn.out_proj.bias)

        self.film_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, pixel_ch * 2, 1) for _ in range(self.n_sources)
        ])
        self.add_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, pixel_ch, 1) for _ in range(self.n_sources)
        ])
        self.gate_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, 1, 1) for _ in range(self.n_sources)
        ])
        for module_list in (self.film_convs, self.add_convs, self.gate_convs):
            for conv in module_list:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def _pos_tokens(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords)

    def _refine_sources(self, ctx_list):
        b, _, h, w = ctx_list[0].shape
        pos = self._pos_tokens(h, w, ctx_list[0].device, ctx_list[0].dtype)
        seqs = []
        for i, ctx in enumerate(ctx_list):
            tokens = ctx.flatten(2).transpose(1, 2)
            tokens = tokens + pos
            tokens = tokens + self.modality_embed[i].view(1, 1, -1)
            seqs.append(tokens)
        x = torch.cat(seqs, dim=1)

        x_norm = self.attn_norm(x)
        with torch.amp.autocast("cuda", enabled=False):
            attn_out, _ = self.cross_source_attn(
                x_norm.float(), x_norm.float(), x_norm.float(),
                need_weights=False,
            )
        # Zero-init out_proj => attn_out ~ 0 at init => refined ~ ctx_list[i]
        chunks = attn_out.to(x.dtype).split(h * w, dim=1)
        refined = []
        for i, ck in enumerate(chunks):
            delta = ck.transpose(1, 2).reshape(b, self.ctx_ch, h, w)
            refined.append(ctx_list[i] + delta)
        return refined

    def forward(self, F_pixel, token):
        """
        F_pixel: (B, pixel_ch, H, W)
        token:   (B, n_sources * token_source_ch, h, w)  [e.g. 4x768 at 16x16]
        returns: (B, pixel_ch, H, W)
        """
        H, W = F_pixel.shape[-2:]
        parts = token.float().split(self.token_source_ch, dim=1)
        with torch.amp.autocast("cuda", enabled=False):
            ctx_list = []
            for i, src in enumerate(parts):
                if self.token_calibs is not None:
                    src = self.token_calibs[i](src)
                src = src.clamp(-self._TOKEN_INPUT_CLAMP, self._TOKEN_INPUT_CLAMP)
                ctx = self.token_projs[i](src).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
                ctx_list.append(ctx)

            refined = self._refine_sources(ctx_list)
            refined = [r.clamp(-self._CTX_CLAMP, self._CTX_CLAMP) for r in refined]

            F_p_f = F_pixel.float()
            delta = torch.zeros_like(F_p_f)
            for i, ctx in enumerate(refined):
                ctx_up = F.interpolate(
                    ctx, size=(H, W), mode="bilinear", align_corners=False
                )
                gamma, beta = self.film_convs[i](ctx_up).chunk(2, dim=1)
                gamma = gamma.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                beta = beta.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                g = torch.sigmoid(self.gate_convs[i](ctx_up))
                add = self.add_convs[i](ctx_up).clamp(-self._ADD_CLAMP, self._ADD_CLAMP)
                delta = delta + g * (gamma * F_p_f + beta + add)

            out = (F_p_f + delta).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
        return out.to(F_pixel.dtype)
