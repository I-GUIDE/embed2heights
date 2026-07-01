"""Exponential moving average of model weights.

EMA(theta_t) = decay * EMA(theta_{t-1}) + (1 - decay) * theta_t

Decay 0.999 averages over ~1000 steps; 0.9995 over ~2000. Updated AFTER each
optimizer step (not per grad-accum micro-batch). EMA weights are stored
separately from the live model and saved to model_ema.pth; predict.py picks
them up via --model-path. Non-float buffers (BN counters, integer ids) are
copied straight through.
"""

import torch


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        live_sd = self._live_state_dict(model)
        self.shadow = {k: v.detach().clone() for k, v in live_sd.items()}

    @staticmethod
    def _live_state_dict(model):
        if isinstance(model, torch.nn.DataParallel):
            return model.module.state_dict()
        return model.state_dict()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in self._live_state_dict(model).items():
            if k not in self.shadow:
                continue
            shadow_v = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow_v.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                shadow_v.copy_(v.detach())

    def state_dict(self):
        return self.shadow
