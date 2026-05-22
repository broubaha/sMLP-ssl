import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed=1):
    """Sets numpy and PyTorch seeds for reproducible weight initialisation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────

def mish(x):
    """
    Mish activation: x * tanh(softplus(x)).
    Self-regularised non-monotonic activation (Misra, 2019).
    Reference: https://arxiv.org/abs/1908.08681
    Not used in the main forward pass; kept for experimentation.
    """
    return x * torch.tanh(F.softplus(x))


def ACS(x, n_neurons=3):
    """
    Hard Local Winner-Take-All (ACS = Adaptive Competitive Selection).

    Divides the input into non-overlapping blocks of size n_neurons.
    Within each block, only the neuron with the maximum pre-activation
    retains its value; all others are set to zero. This is the strict
    (binary) version used in the sMLP intermediate layer.

    Args:
        x         : input tensor of shape (B, L)
        n_neurons : block size k1

    Returns:
        Tensor of same shape with losers zeroed out.
    """
    view = x.view(-1, x.shape[1] // n_neurons, n_neurons)  # (B, n_blocks, k1)
    max_values, _ = view.max(dim=2, keepdim=True)
    view *= (view >= max_values)   # zero out sub-maximal neurons
    return view.view_as(x)


def Soft_ACS(x, n_neurons=2, alpha=0.5):
    """
    Soft Local Winner-Take-All (Soft-ACS).

    A relaxed version of ACS where losing neurons are not fully silenced
    but retain a fraction alpha of their pre-activation value. This
    preserves gradient flow for all neurons during backpropagation,
    avoiding dead-neuron issues that can arise with strict WTA.

    The competition rule is:
        winner → value * 1.0   (unchanged)
        losers → value * alpha  (attenuated)

    Setting alpha=0 recovers strict ACS (hard WTA).
    Setting alpha=1 recovers a standard linear layer (no competition).

    This is the activation used in the sMLP intermediate layer (pc1)
    in our CIFAR/STL-10 experiments.

    Args:
        x         : input tensor of shape (B, L)
        n_neurons : block size k1
        alpha     : attenuation factor for losing neurons in [0, 1]

    Returns:
        Tensor of same shape with losers attenuated by alpha.
    """
    batch_size   = x.shape[0]
    num_features = x.shape[1]

    # Reshape into blocks: (B, n_blocks, k1)
    view = x.view(batch_size, num_features // n_neurons, n_neurons)

    # Identify winners and losers within each block
    max_values, _ = view.max(dim=2, keepdim=True)
    mask_winners  = (view >= max_values).float()          # 1 for winner, 0 for losers
    mask_losers   = (view < max_values).float() * alpha   # alpha for losers, 0 for winner

    # Apply soft WTA: winner at 100%, losers at alpha%
    view = view * (mask_winners + mask_losers)

    return view.view(batch_size, num_features)


# ─────────────────────────────────────────────
# PSBatchNorm2d (optional, not used in main forward)
# ─────────────────────────────────────────────

class PSBatchNorm2d(nn.BatchNorm2d):
    """
    Positive-Shift BatchNorm: adds a small positive constant alpha after
    standard BN to prevent filter collapse in very deep networks.
    Reference: https://arxiv.org/abs/2001.11216
    Not used in the main forward pass; kept for experimentation.
    """
    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


# ─────────────────────────────────────────────
# WideResNet building blocks
# ─────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    WideResNet basic residual block.

    Structure: BN → LeakyReLU → Conv3×3 → BN → LeakyReLU → (Dropout) → Conv3×3
    with a skip connection. When in_planes ≠ out_planes, a 1×1 conv shortcut
    is used to match dimensions.

    activate_before_residual=True: applies BN+ReLU to x before the residual
    path, used in the first block of the network where x has not yet been
    normalised.
    """
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0,
                 activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1   = nn.BatchNorm2d(in_planes,  momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate    = drop_rate
        self.equalInOut   = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                      padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))   # pre-activate x for the first block
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """
    Stacks nb_layers BasicBlocks. The first block uses the specified stride
    (for downsampling); subsequent blocks use stride=1.
    """
    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
                 drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride,
            drop_rate, activate_before_residual
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(
                i == 0 and in_planes or out_planes,  # in_planes only for first block
                out_planes,
                i == 0 and stride or 1,              # stride only for first block
                drop_rate, activate_before_residual
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


# ─────────────────────────────────────────────
# WideResNet + sMLP head
# ─────────────────────────────────────────────

class WideResNet(nn.Module):
    """
    WideResNet encoder with a two-layer sparse MLP (sMLP) classification head.

    Encoder: standard WideResNet-28-k (k = model_width).
        - 3 groups of BasicBlocks with widths [16k, 32k, 64k]
        - Global average pooling → feature vector of size 64k

    sMLP decoder (our contribution, replaces the standard dense FC layer):
        - pc1: sparse linear (channels[3] → L1), pruned with sparsity s1
               followed by BN (bn3) and Soft-ACS competition
        - pc2: sparse linear (L1 → num_classes × n_blocks), pruned with s2
               output organised as n_blocks blocks of num_classes logits each

    Baseline mode (use_fc=True):
        - Replaces the sMLP head with a standard dense FC layer (fc)
        - Used to reproduce the InfoMatch baseline

    Initialisation strategy ("double initialisation"):
        1. pc1 and pc2 are initialised with small uniform values before pruning
           to avoid large pre-activations at the start of training.
        2. After pruning, all Conv2d layers are re-initialised with Kaiming normal,
           BN layers with constant 1/0, and Linear layers with Xavier normal.
           This second pass overwrites the uniform init of pc1/pc2 — intentional,
           as Xavier normal provides better scaling for the surviving weights.

    Args:
        args      : training arguments (model_depth, model_width, num_classes,
                    n_blocks, l1, k1, s1, s2, alpha, use_fc, seed)
        drop_rate : dropout rate within BasicBlocks (default 0)
    """

    def __init__(self, args, drop_rate=0.0):
        # Fix seed before weight drawing so that sparse masks are reproducible
        set_seed(args.seed)
        super(WideResNet, self).__init__()

        # Channel widths for the three WideResNet groups
        channels = [16,
                    16 * args.model_width,
                    32 * args.model_width,
                    64 * args.model_width]

        assert (args.model_depth - 4) % 6 == 0, \
            "model_depth must satisfy (depth - 4) % 6 == 0"
        n = (args.model_depth - 4) / 6   # number of blocks per group

        # sMLP hyperparameters (with fallback defaults for compatibility)
        try:
            l1 = args.l1
        except AttributeError:
            l1 = 128
        try:
            self.k1 = args.k1
        except AttributeError:
            self.k1 = 2

        # ── Encoder ──────────────────────────────────────────────────────────
        self.conv1  = nn.Conv2d(3, channels[0], kernel_size=3, stride=args.first_stride,
                                padding=1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], BasicBlock, 1,
                                   drop_rate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, channels[1], channels[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], BasicBlock, 2, drop_rate)
        self.bn1    = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu   = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # ── sMLP decoder ─────────────────────────────────────────────────────
        # Baseline: single dense FC head (InfoMatch reference)
        self.fc  = nn.Linear(channels[3], args.num_classes)

        # pc1: sparse intermediate layer
        self.pc1 = nn.Linear(channels[3], l1)
        self.bn3 = nn.BatchNorm1d(l1)   # BN applied before Soft-ACS competition

        # pc2: sparse output layer — produces n_blocks × num_classes logits
        # organised as [block_0_logits | block_1_logits | ... | block_{B2-1}_logits]
        self.pc2 = nn.Linear(l1, args.num_classes * args.n_blocks)

        # Small uniform init for pc1/pc2 before pruning (avoids large activations)
        nn.init.uniform_(self.pc1.weight, a=-0.005, b=0.005)
        nn.init.uniform_(self.pc2.weight, a=-0.01,  b=0.01)

        self.alpha  = args.alpha
        self.use_fc = args.use_fc

        # ── Sparse connectivity ───────────────────────────────────────────────
        # L1-unstructured pruning: zeros out the s1 (resp. s2) fraction of
        # weights with smallest absolute value. The mask is fixed after init
        # (no pruning schedule — structural sparsity throughout training).
        prune.l1_unstructured(self.pc1, name='weight', amount=args.s1)
        prune.l1_unstructured(self.pc2, name='weight', amount=args.s2)

        self.channels = channels[3]

        # ── Second initialisation pass ────────────────────────────────────────
        # Overwrites all Conv2d, BN, and Linear layers (including pc1/pc2)
        # with standard initialisations. The pruning masks are preserved.
        # Note: this overwrites the small uniform init of pc1/pc2 with
        # Xavier normal — intentional for better gradient scaling.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass.

        Encoder: Conv → Block1 → Block2 → Block3 → BN+ReLU → AvgPool → flatten
        Decoder (sMLP): pc1 → BN → Soft-ACS → pc2  (or fc in baseline mode)

        Returns:
            output : logits of shape (B, num_classes) in baseline mode, or
                     (B, num_classes * n_blocks) in sMLP mode.
                     In sMLP mode, the output is organised as n_blocks
                     consecutive blocks of num_classes logits.
        """
        # Encoder
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)   # global average pooling → (B, C, 1, 1)
        out = out.view(out.size(0), -1)        # flatten → (B, channels[3])

        # Decoder
        if self.use_fc:
            # Baseline: standard dense classification head
            output = self.fc(out)
        else:
            # sMLP head
            out    = self.pc1(out)                              # sparse linear: (B, L1)
            out    = self.bn3(out)                              # BN before competition
            out    = Soft_ACS(out, n_neurons=self.k1,
                               alpha=self.alpha)                # soft WTA: (B, L1)
            output = self.pc2(out)                              # sparse linear: (B, n_blocks * C)

        return output


# ─────────────────────────────────────────────
# Builder and utilities
# ─────────────────────────────────────────────

def build_wideresnet(args):
    """Instantiates a WideResNet with sMLP head from training arguments."""
    return WideResNet(args)


def count_parameters_in_millions(model):
    """Returns the total number of trainable parameters in millions."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1_000_000
