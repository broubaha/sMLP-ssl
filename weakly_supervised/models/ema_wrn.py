from copy import deepcopy
import torch


# ─────────────────────────────────────────────
# Exponential Moving Average (EMA) model wrapper
# ─────────────────────────────────────────────

class ModelEMA(object):
    """
    Maintains an Exponential Moving Average (EMA) copy of a model.

    The EMA model is used exclusively for evaluation. Its weights are a
    slow-moving average of the live model weights, which typically produces
    better generalisation than the instantaneous weights, especially in SSL
    where training signal is noisy (pseudo-labels).

    The EMA update rule for each parameter θ is:
        θ_ema ← decay × θ_ema + (1 - decay) × θ_live

    With decay=0.999, the EMA model effectively averages over ~1000 past
    iterations, acting as a smooth ensemble of recent model snapshots.

    Non-learnable buffers (e.g. BatchNorm running_mean and running_var)
    are copied directly from the live model without EMA smoothing, so
    that BN statistics stay up to date with the current data distribution.

    The EMA model is frozen (requires_grad=False) and kept in eval() mode
    at all times — it never receives gradient updates directly.

    Args:
        args  : training arguments (unused here, kept for API compatibility)
        model : the live model being trained
        ema   : a second model instance of the same architecture,
                used to store the EMA weights (initialised from model)
        decay : EMA decay rate (typically 0.999)
    """

    def __init__(self, args, model, ema, decay):
        # Initialise EMA weights from the current live model state
        self.ema = ema
        self.ema.load_state_dict(model.state_dict())
        self.ema.cuda()

        # Keep EMA model in eval mode throughout training:
        # - BatchNorm uses its running statistics (not batch statistics)
        # - Dropout is disabled
        self.ema.eval()

        self.decay = decay

        # Check whether the EMA model is wrapped in nn.DataParallel / DDP.
        # If the live model is DDP-wrapped (has a 'module' attribute) but the
        # EMA model is not, parameter keys must be prefixed with 'module.'
        # when reading from the live model's state dict.
        self.ema_has_module = hasattr(self.ema, 'module')

        # Collect parameter and buffer key names from the EMA model.
        # param_keys  : learnable parameters (updated with EMA smoothing)
        # buffer_keys : non-learnable buffers (copied directly, e.g. BN stats)
        self.param_keys  = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]

        # Freeze all EMA parameters — no gradient computation needed
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        Updates the EMA model weights after each training step.

        For learnable parameters: EMA smoothing
            θ_ema ← decay × θ_ema + (1 - decay) × θ_live

        For non-learnable buffers (BN running stats): direct copy
            buffer_ema ← buffer_live

        The DDP key prefix ('module.') is handled transparently:
        if the live model is DDP-wrapped and the EMA model is not,
        the prefix is added when reading from the live state dict.

        Args:
            model : the live model after its latest parameter update
        """
        # Determine whether to prepend 'module.' when reading live model keys
        # (happens when model is DDP-wrapped but ema is not)
        needs_module = hasattr(model, 'module') and not self.ema_has_module

        with torch.no_grad():
            msd = model.state_dict()  # live model state dict
            esd = self.ema.state_dict()  # EMA model state dict

            # ── Learnable parameters: EMA update ─────────────────────────────
            for k in self.param_keys:
                j = ('module.' + k) if needs_module else k
                model_v = msd[j].detach()
                ema_v   = esd[k]
                # EMA formula: θ_ema ← decay × θ_ema + (1 - decay) × θ_live
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            # ── Non-learnable buffers: direct copy ───────────────────────────
            # BN running_mean and running_var are copied directly so that the
            # EMA model reflects the current data statistics without lag.
            for k in self.buffer_keys:
                j = ('module.' + k) if needs_module else k
                esd[k].copy_(msd[j])
