import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from dataset.cifar_cutmix_w import DATASET_GETTERS
from utils.misc import AverageMeter
from dataset.mix_1 import mixup_soft, mixup_hard, cutmix_soft, cutmix_hard

# ─────────────────────────────────────────────
# Distributed training setup (Jean Zay / Odyssey HPC clusters)
# idr_torch provides rank, local_rank, and world_size for SLURM-based DDP.
# See: http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-torch-multi.html
# ─────────────────────────────────────────────
import idr_torch

# Path to dataset on local storage (adjust to your cluster's scratch directory)
data_dir = './data'

logger = logging.getLogger(__name__)
best_acc = 0


# ─────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    """
    Saves the current training state to disk.
    If this is the best model so far, also copies it to 'model_best.pth.tar'.
    """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(args):
    """Sets all random seeds for full reproducibility across CPU, GPU, and cuDNN."""
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# Learning rate scheduler
# ─────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=7./16., last_epoch=-1):
    """
    Cosine annealing LR schedule with linear warmup.

    During warmup (first num_warmup_steps steps): LR increases linearly from 0 to base LR.
    After warmup: LR follows a cosine decay down to 0.

    num_cycles=7/16 gives a partial cosine cycle (does not reach 0 at the end),
    which has been found empirically effective for SSL training.
    """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


# ─────────────────────────────────────────────
# Interleaving utilities for mixed batch forward pass
# ─────────────────────────────────────────────

def interleave(x, size):
    """
    Reshuffles a concatenated batch so that labeled and unlabeled samples
    are interleaved across the batch dimension. This ensures that each
    mini-batch seen by BatchNorm contains a mix of both types, avoiding
    statistics bias from processing them sequentially.

    Args:
        x    : concatenated tensor of shape (total_batch, ...)
        size : number of interleave groups (= 1 + mu * n_views)

    Returns:
        Interleaved tensor of same shape.
    """
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    """Reverses the interleave operation to restore original batch ordering."""
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# ─────────────────────────────────────────────
# Adaptive threshold estimation (FreeMatch / SoftMatch style)
# ─────────────────────────────────────────────

@torch.no_grad()
def cal_time_p_and_p_model(logits_x_ulb_w, time_p, p_model, label_hist):
    """
    Updates three exponential moving averages used for adaptive thresholding:

        time_p      : global mean confidence (scalar EMA) — used as base threshold
        p_model     : per-class mean confidence (vector EMA) — used to compute
                      class-specific thresholds that compensate for class imbalance
        label_hist  : per-class prediction frequency (vector EMA) — tracks how
                      often each class is selected as pseudo-label

    All three are updated with momentum 0.999 (slow-moving averages).
    The per-sample threshold is: time_p * p_model[predicted_class] / max(p_model)

    Args:
        logits_x_ulb_w : raw logits from the weak augmentation view, shape (B, C)
        time_p, p_model, label_hist : current EMA values (None on first call)

    Returns:
        Updated (time_p, p_model, label_hist)
    """
    prob_w = torch.softmax(logits_x_ulb_w, dim=1)
    max_probs, max_idx = torch.max(prob_w, dim=-1)

    # Scalar EMA of mean confidence across the batch
    time_p = max_probs.mean() if time_p is None \
        else time_p * 0.999 + max_probs.mean() * 0.001

    # Vector EMA of per-class mean probability
    p_model = torch.mean(prob_w, dim=0) if p_model is None \
        else p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001

    # Vector EMA of per-class prediction frequency
    hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype)
    hist = hist / hist.sum()
    label_hist = hist if label_hist is None \
        else label_hist * 0.999 + hist * 0.001

    return time_p, p_model, label_hist


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='sMLP semi-supervised training on CIFAR/STL-10')

    # --- Infrastructure ---
    parser.add_argument('--gpu-id',      default='0',   type=int,   help='CUDA device id')
    parser.add_argument('--num-workers', type=int,      default=4,  help='DataLoader worker processes')
    parser.add_argument('--local_rank',  type=int,      default=-1, help='DDP local rank (set automatically by SLURM)')
    parser.add_argument('--no-progress', action='store_true',       help='disable tqdm progress bars')
    parser.add_argument('--amp',         action='store_true',       help='enable NVIDIA apex mixed-precision training')
    parser.add_argument('--opt_level',   type=str,      default='O1', help='apex AMP optimisation level')

    # --- Dataset ---
    parser.add_argument('--dataset',      default='cifar100', type=str,
                        choices=['cifar10', 'cifar100', 'SVHN', 'STL10'], help='dataset name')
    parser.add_argument('--num-labeled',  type=int, default=400,   help='number of labelled samples')
    parser.add_argument('--seed-labeled', default=303, type=int,   help='seed used to draw the labelled split')
    parser.add_argument('--expand-labels', action='store_true',    help='expand labels to match eval steps')

    # --- Architecture ---
    parser.add_argument('--arch',         default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],      help='encoder architecture')
    parser.add_argument('--first-stride', default=1, type=int,    help='stride of first conv block (2 for STL-10)')

    # --- sMLP head hyperparameters ---
    parser.add_argument('--n-blocks',  default=2,    type=int,   help='number of output blocks B2 (repetition coding)')
    parser.add_argument('--s1',        default=0.92, type=float, help='sparsity of the intermediate layer')
    parser.add_argument('--s2',        default=0.92, type=float, help='sparsity of the output layer')
    parser.add_argument('--l1',        default=255,  type=int,   help='size of the intermediate layer L1')
    parser.add_argument('--k1',        default=3,    type=int,   help='block size for LWTA in intermediate layer')
    parser.add_argument('--use-fc',    action='store_true', default=False,
                        help='replace sMLP head with a standard dense FC layer (baseline mode)')
    parser.add_argument('--baseline',  action='store_true', default=False,
                        help='shortcut flag: enables use-fc and sets n-blocks=1 for the InfoMatch baseline')

    # --- Optimisation ---
    parser.add_argument('--total-steps',  default=1024*300, type=int,  help='total training iterations')
    parser.add_argument('--eval-step',    default=1024,     type=int,  help='iterations per evaluation epoch')
    parser.add_argument('--start-epoch',  default=0,        type=int,  help='epoch to resume from')
    parser.add_argument('--batch-size',   default=64,       type=int,  help='labelled batch size per GPU')
    parser.add_argument('--lr',           default=0.03,     type=float, help='initial learning rate')
    parser.add_argument('--warmup',       default=0,        type=float, help='LR warmup steps (unlabelled-data based)')
    parser.add_argument('--warmbatch',    default=0,        type=float, help='supervised warmup iterations')
    parser.add_argument('--wdecay',       default=1e-3,     type=float, help='weight decay (L2 regularisation)')
    parser.add_argument('--nesterov',     action='store_true', default=True, help='use Nesterov SGD momentum')
    parser.add_argument('--mu',           default=7,        type=int,  help='unlabelled-to-labelled batch size ratio')
    parser.add_argument('--lambda-u',     default=2,        type=float, help='weight of the unsupervised loss')
    parser.add_argument('--T',            default=1,        type=float, help='softmax temperature for pseudo-labels')
    parser.add_argument('--threshold',    default=0.95,     type=float, help='fixed confidence threshold (not used with adaptive)')
    parser.add_argument('--lam-sim',      default=0.2,      type=float, help='weight of the inter-block cosine alignment loss')
    parser.add_argument('--alpha',        default=0.0,                  help='MixUp alpha (unused, kept for compatibility)')
    parser.add_argument('--is-EL',        default=1,        type=int,   help='1: use EMA logits for thresholding, 0: use mean logits')
    parser.add_argument('--seed',         default=1,        type=int,   help='global random seed')

    # --- EMA ---
    parser.add_argument('--use-ema',   action='store_true', default=True,  help='maintain an EMA copy of the model for evaluation')
    parser.add_argument('--ema-decay', default=0.999,       type=float,    help='EMA decay rate')

    # --- I/O ---
    parser.add_argument('--out',    default='result', help='output directory for checkpoints and logs')
    parser.add_argument('--resume', default='',       type=str, help='path to checkpoint to resume from')

    args = parser.parse_args()

    # ── Output directory naming ───────────────────────────────────────────────
    # Baseline: single dense FC head, InfoMatch-style
    # Ours:     sMLP head with configurable sparsity, blocks, and alignment loss
    if args.baseline:
        args.use_fc   = True
        args.n_blocks = 1
        suffix = 'EL' if args.is_EL else 'mean'
        args.out = f"result/{args.dataset}@{args.num_labeled}/{args.seed}_Baseline_{suffix}_ody_bn3"
    else:
        suffix = 'EL' if args.is_EL else 'mean'
        args.out = (f"result/{args.dataset}@{args.num_labeled}/"
                    f"{args.s1}_{args.s2}_{args.l1}_{args.n_blocks}_"
                    f"{args.lam_sim}_{args.k1}_{args.alpha}/{args.seed}_Our_{suffix}_ody_bn3")

    # Auto-resume if a checkpoint already exists in the output directory
    checkpoint_path = os.path.join(args.out, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        args.resume = checkpoint_path

    global best_acc

    # ── DDP initialisation (SLURM / idr_torch) ───────────────────────────────
    args.local_rank = idr_torch.local_rank
    args.world_size = idr_torch.size
    args.n_gpu      = torch.cuda.device_count()

    torch.cuda.set_device(args.local_rank)
    device      = torch.device("cuda", args.local_rank)
    args.device = device
    device_prop = torch.cuda.get_device_properties(args.device)

    print(f"Distributed training: rank {args.local_rank}/{args.world_size}, "
          f"device={device}, model={device_prop.name}, "
          f"VRAM={math.ceil(device_prop.total_memory/(1024**3))}G")
    if args.local_rank == 0:
        print(f"Total batch size: {args.batch_size} samples/GPU × "
              f"{args.world_size} GPUs = {args.batch_size * args.world_size}")

    # Initialise the NCCL process group for all-reduce operations
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=idr_torch.size,
        rank=idr_torch.rank,
        device_id=device
    )

    # ── Logging (rank 0 only) ─────────────────────────────────────────────────
    if args.local_rank in [-1, 0]:
        log_dir  = './log'
        os.makedirs(log_dir, exist_ok=True)
        log_file = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_path = os.path.join(log_dir, log_file + '.log')
        open(log_path, 'w').close()  # create empty log file
        logging.basicConfig(
            filename=log_path,
            format="%(asctime)s: %(levelname)s: %(name)s: %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        logger.info(dict(args._get_kwargs()))
        print("Training setup:", dict(args._get_kwargs()))
        print("Log file:", log_file + '.log')

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)

    # ── Dataset-specific architecture config ─────────────────────────────────
    # WRN-28-2 for CIFAR-10 / STL-10 (smaller class space)
    # WRN-28-8 for CIFAR-100        (larger class space, wider network)
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.first_stride = 1
            args.model_depth  = 28
            args.model_width  = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth       = 28
            args.model_width       = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.wdecay      = 1e-3
        if args.arch == 'wideresnet':
            args.first_stride = 1
            args.model_depth  = 28
            args.model_width  = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth       = 29
            args.model_width       = 64

    elif args.dataset == 'SVHN':
        args.num_classes = 10
        args.warmbatch   = 2048  # SVHN benefits from a short supervised warm-up
        if args.arch == 'wideresnet':
            args.first_stride = 1
            args.model_depth  = 28
            args.model_width  = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth       = 28
            args.model_width       = 4

    elif args.dataset == 'STL10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            # STL-10 images are 96×96; first stride=2 avoids excessive feature map size
            args.first_stride = 2
            args.model_depth  = 28
            args.model_width  = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth       = 28
            args.model_width       = 4

    # Barrier: rank 0 downloads/prepares data before other ranks access it
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, data_dir
    )

    # Special case: 10 labels for CIFAR-10 uses fixed FixMatch index sets
    # to ensure comparability with published baselines
    if args.num_labeled == 10 and args.dataset == 'cifar10':
        fixmatch_index = [
            [7408, 8148, 9850, 10361, 33949, 36506, 37018, 45044, 46443, 47447],
            [5022, 8193, 8902, 9601, 25226, 26223, 34089, 35186, 40595, 48024],
            [7510, 13186, 14043, 21305, 22805, 31288, 34508, 40470, 41493, 45506],
            [9915, 9978, 16631, 19915, 28008, 35314, 35801, 36149, 39215, 42557],
            [6695, 14891, 19726, 22715, 23999, 34230, 46511, 47457, 49181, 49397],
            [12830, 20293, 26835, 30517, 30898, 31061, 43693, 46501, 47310, 48517],
            [1156, 11501, 19974, 21963, 32103, 42189, 46789, 47690, 48229, 48675],
            [4255, 6446, 8580, 11759, 12598, 29349, 29433, 33759, 35345, 38639]
        ]
        index = fixmatch_index[-args.seed - 1]
        print("Using fixed 10-label split for CIFAR-10 (FixMatch protocol)")
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
            args, data_dir, index
        )

    if args.local_rank == 0:
        print("weight decay:", args.wdecay)
        torch.distributed.barrier()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # Use DistributedSampler for DDP; RandomSampler for single-GPU runs.
    train_sampler = RandomSampler if args.local_rank == -1 else \
        partial(DistributedSampler, num_replicas=idr_torch.size,
                rank=idr_torch.rank, shuffle=True)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        shuffle=(train_sampler == RandomSampler),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True   # drop incomplete last batch to keep consistent BN statistics
    )

    # Unlabelled loader uses a batch size mu× larger than the labelled loader
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        shuffle=(train_sampler == RandomSampler),
        batch_size=args.batch_size * args.mu,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),  # deterministic order for evaluation
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Approximate parameter count of the sMLP decoder (for sanity check)
    print(f"sMLP decoder non-zero params ≈ "
          f"{512*args.l1*(1-args.s2) + args.l1*args.n_blocks*args.num_classes*(1-args.s2):.0f}")

    # ── Model creation ────────────────────────────────────────────────────────
    def create_model(args):
        """
        Instantiates the encoder + sMLP head.

        wideresnet_e: our variant with the sMLP decoder and double initialisation.
        The EMA model (ema) is a separate instance kept in sync via exponential
        moving average of the live model weights — it is used for evaluation only.
        """
        if args.arch == 'wideresnet':
            import models.wideresnet_e as models  # sMLP variant with double init
            model = models.build_wideresnet(args)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(
                cardinality=args.model_cardinality,
                depth=args.model_depth,
                width=args.model_width,
                num_classes=args.num_classes
            )
        if args.local_rank in [-1, 0]:
            logger.info("Total params: {:.2f}M".format(
                sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    model = create_model(args)
    ema   = create_model(args)  # second instance for EMA tracking

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Weight decay is not applied to bias terms and BatchNorm parameters,
    # following standard practice in SSL literature.
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler   = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    # EMA model: maintains a slow-moving average of model weights.
    # At test time, the EMA model typically generalises better than the live model.
    if args.use_ema:
        from models.ema_wrn import ModelEMA
        ema_model = ModelEMA(args, model, ema, args.ema_decay)

    args.start_epoch = 0

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume:
        if args.local_rank in [-1, 0]:
            print("==> Resuming from checkpoint..")
            logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: checkpoint not found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc         = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # ── DDP wrapping ──────────────────────────────────────────────────────────
    # bn3 (the BN before LWTA in the sMLP head) must be converted to
    # SyncBatchNorm before DDP wrapping so that statistics are synchronised
    # across all GPUs. Other BN layers (in the encoder) are left as-is
    # because they use BatchNorm2d which is handled correctly by DDP.
    if args.local_rank != -1:
        model.bn3 = torch.nn.SyncBatchNorm(
            num_features=model.bn3.num_features,
            eps=model.bn3.eps,
            momentum=model.bn3.momentum,
            affine=model.bn3.affine,
            track_running_stats=model.bn3.track_running_stats
        ).to(args.device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )

    if args.local_rank in [-1, 0]:
        logger.info("***** Running training *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
        logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


# ─────────────────────────────────────────────
# Supervised warm-up (optional)
# ─────────────────────────────────────────────

def warmup(args, labeled_trainloader, test_loader, model, optimizer, scheduler):
    """
    Optional supervised pre-training phase (used for SVHN).

    Trains on labelled data only for args.warmbatch iterations using standard
    cross-entropy loss. At the end, computes initial estimates of time_p,
    p_model, and label_hist from the test set to seed the adaptive thresholds
    before the main SSL loop begins.

    Returns:
        time_p, p_model, label_hist : initial threshold EMA values
    """
    if args.amp:
        from apex import amp
    global best_acc

    if args.world_size > 1:
        labeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    model.train()

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    if not args.no_progress:
        p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])

    end = time.time()
    for i in range(args.warmbatch):
        try:
            inputs_x, targets_x = next(labeled_iter)
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_trainloader.sampler.set_epoch(labeled_epoch)
            labeled_iter    = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_iter)

        inputs_x  = inputs_x.to(args.device)
        data_time.update(time.time() - end)

        inputs   = interleave(inputs_x, 1).to(args.device)
        targets_x = targets_x.to(args.device)
        logits   = model(inputs)
        logits   = de_interleave(logits, 1)

        loss = F.cross_entropy(logits, targets_x, reduction='mean')

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        losses.update(loss.item())
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if not args.no_progress:
            p_bar.set_description(
                f"WarmIter: {i+1:4}/2048. LR: {scheduler.get_last_lr()[0]:.4f}. "
                f"Data: {data_time.avg:.3f}s. Batch: {batch_time.avg:.3f}s. "
                f"Loss: {losses.avg:.4f}."
            )
            p_bar.update()

    if args.local_rank in [-1, 0]:
        logger.info(f"Warmup done — lr:{scheduler.get_last_lr()[0]:.4f}, "
                    f"loss:{losses.avg:.4f}")

    if not args.no_progress:
        p_bar.close()

    # Compute initial threshold estimates from the test set
    model.eval()
    if args.local_rank in [-1, 0]:
        probs = []
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(args.device)
                outputs = model(inputs)
                probs.append(outputs.softmax(dim=-1))
        probs = torch.cat(probs)
        max_probs, max_idx = torch.max(probs, dim=-1)
        time_p     = max_probs.mean()
        p_model    = torch.mean(probs, dim=0)
        label_hist = torch.bincount(max_idx, minlength=probs.shape[1]).to(probs.dtype)
        label_hist = label_hist / label_hist.sum()

    return time_p, p_model, label_hist


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    """
    Main SSL training loop.

    At each iteration:
        1. Draw a labelled batch (inputs_x, targets_x).
        2. Draw an unlabelled batch: (weak, cutmix_weak, strong1, strong2).
        3. Concatenate and interleave all views for a single forward pass.
        4. For each output block i in [0, n_blocks):
              a. Compute supervised CE loss Lx on the labelled view.
              b. Update adaptive thresholds (time_p, p_model, label_hist).
              c. Generate pseudo-labels from the weak view.
              d. Apply per-sample thresholding mask.
              e. Compute unsupervised loss Lu:
                   - CutMix CE (soft label mixing)
                   - Strong view CE (×0.25 each for s1 and s2)
                   - MSE consistency between s1 and s2 normalised logits
        5. If n_blocks > 1, add inter-block cosine alignment loss L_align
           (subtracted because it is maximised, not minimised).
        6. Total loss: (Lx + 2*Lu) / n_blocks - lam_sim * L_align
        7. Update EMA model, save checkpoint every epoch.

    The CutMix augmentation on the weak view injects additional diversity
    by mixing pairs of unlabelled images and interpolating their pseudo-labels.
    """
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end       = time.time()

    if args.world_size > 1:
        labeled_epoch   = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter   = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    # Initialise adaptive threshold EMAs
    if args.resume:
        checkpoint   = torch.load(args.resume)
        time_p       = [t.to(args.device) for t in checkpoint['time_p']]
        p_model      = [t.to(args.device) for t in checkpoint['p_model']]
        label_hist   = [t.to(args.device) for t in checkpoint['label_hist']]
    elif args.warmbatch > 0:
        print('Warm-up stage')
        time_p, p_model, label_hist = warmup(
            args, labeled_trainloader, test_loader, model, optimizer, scheduler
        )
    else:
        # Default initialisation: uniform distribution over classes
        p_model    = [(torch.ones(args.num_classes) / args.num_classes).to(args.device)
                      for _ in range(args.n_blocks)]
        label_hist = [(torch.ones(args.num_classes) / args.num_classes).to(args.device)
                      for _ in range(args.n_blocks)]
        time_p     = [p_model[i].mean() for i in range(args.n_blocks)]

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses     = AverageMeter()
        losses_x   = AverageMeter()
        losses_u   = AverageMeter()
        mask_probs = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        for batch_idx in range(args.eval_step):

            # ── Fetch labelled batch ──────────────────────────────────────────
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter         = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            # ── Fetch unlabelled batch ────────────────────────────────────────
            # Each unlabelled item provides: weak view, two strong views,
            # and CutMix parameters (bbox_w, lam_w) pre-computed by the dataset.
            try:
                (inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w), _ = next(unlabeled_iter)

            inputs_x, inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w = (
                inputs_x.to(args.device),   inputs_u_w.to(args.device),
                inputs_u_s1.to(args.device), inputs_u_s2.to(args.device),
                bbox_w.to(args.device),      lam_w.to(args.device)
            )

            # Random permutation for CutMix pairing among unlabelled samples
            indices  = torch.randperm(inputs_u_w.size(0))
            cutmix_w = cutmix_hard(inputs_u_w, indices, bbox_w)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]

            # ── Single forward pass over all views (interleaved for BN) ───────
            # Concatenation order: [labeled | weak | cutmix_weak | strong1 | strong2]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, cutmix_w, inputs_u_s1, inputs_u_s2)),
                4 * args.mu + 1
            ).to(args.device)
            del inputs_x, inputs_u_w, cutmix_w, inputs_u_s1, inputs_u_s2

            logits = model(inputs)  # (total_batch, n_blocks * num_classes)

            targets_x = targets_x.to(args.device)
            logits    = de_interleave(logits, 4 * args.mu + 1)

            # Split logits back into their respective views
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_w_cutmix, logits_u_s1, logits_u_s2 = \
                logits[batch_size:].chunk(4)
            del logits

            # ── Per-block loss computation ─────────────────────────────────────
            Lx, Lu, L_align = 0., 0., 0.

            for i in range(args.n_blocks):
                # Block i slice: columns [i*C : (i+1)*C]
                sl = slice(i * args.num_classes, (i + 1) * args.num_classes)

                # Supervised cross-entropy on the labelled view
                Lx += F.cross_entropy(logits_x[:, sl], targets_x, reduction='mean')

                # Update adaptive threshold EMAs for this block
                time_p[i], p_model[i], label_hist[i] = cal_time_p_and_p_model(
                    logits_u_w[:, sl], time_p[i], p_model[i], label_hist[i]
                )

                # Pseudo-labels from the weak view (soft → hard argmax)
                pseudo_label = torch.softmax(logits_u_w[:, sl].detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                # Per-sample adaptive threshold:
                # threshold_i = time_p * (p_model[predicted_class] / max(p_model))
                # Samples below this threshold are masked out (mask=0).
                p_cutoff        = time_p[i]
                p_model_cutoff  = p_model[i] / torch.max(p_model[i])
                threshold       = p_cutoff * p_model_cutoff[targets_u]
                mask            = max_probs.ge(threshold).float()

                # Normalise strong-view logits for MSE consistency term
                pred_u_s1_norm = (
                    (logits_u_s1[:, sl] - logits_u_s1[:, sl].mean(1, keepdim=True))
                    / logits_u_s1[:, sl].std(1, keepdim=True)
                )
                pred_u_s2_norm = (
                    (logits_u_s2[:, sl] - logits_u_s2[:, sl].mean(1, keepdim=True))
                    / logits_u_s2[:, sl].std(1, keepdim=True)
                )

                # Unsupervised loss:
                #   0.5 × CutMix CE (soft label mixing between pseudo-label pairs)
                #   0.25 × CE on strong view 1
                #   0.25 × CE on strong view 2
                #   0.001 × MSE between normalised strong views (consistency)
                Lu += (
                    0.5 * (
                        lam_w * F.cross_entropy(logits_u_w_cutmix[:, sl], targets_u, reduction='none') * mask
                        + (1 - lam_w) * F.cross_entropy(logits_u_w_cutmix[:, sl], targets_u[indices], reduction='none') * mask[indices]
                    ).mean()
                    + 0.25 * (F.cross_entropy(logits_u_s1[:, sl], targets_u, reduction='none') * mask).mean()
                    + 0.25 * (F.cross_entropy(logits_u_s2[:, sl], targets_u, reduction='none') * mask).mean()
                    + 0.001 * F.mse_loss(pred_u_s1_norm, pred_u_s2_norm)
                )

            # ── Inter-block alignment loss ─────────────────────────────────────
            # Cosine similarity between consecutive output blocks is maximised
            # (subtracted from the total loss) to encourage the blocks to produce
            # consistent label estimates on unlabelled data.
            if args.n_blocks > 1:
                num_align_terms = 0
                for j in range(args.n_blocks - 1):
                    L_align += cosine_similarity_loss(
                        logits_u_w[:, j * args.num_classes:(j + 1) * args.num_classes],
                        logits_u_w[:, (j + 1) * args.num_classes:(j + 2) * args.num_classes]
                    )
                    num_align_terms += 1
                # Also align first and last block for n_blocks >= 3
                if args.n_blocks > 2:
                    L_align += cosine_similarity_loss(
                        logits_u_w[:, 0:args.num_classes],
                        logits_u_w[:, (args.n_blocks - 1) * args.num_classes:args.n_blocks * args.num_classes]
                    )
                    num_align_terms += 1
                loss = (Lx + 2.0 * Lu) / args.n_blocks - args.lam_sim * L_align / num_align_terms
            else:
                loss = (Lx + 2.0 * Lu) / args.n_blocks

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

            if not args.no_progress:
                p_bar.set_description(
                    f"{epoch+1}/{args.epochs:4}. Iter: {batch_idx+1:4}/{args.eval_step:4}. "
                    f"LR: {scheduler.get_last_lr()[0]:.4f}. "
                    f"Loss: {losses.avg:.4f}. Loss_x: {losses_x.avg:.4f}. "
                    f"Loss_u: {losses_u.avg:.4f}. Mask: {mask_probs.avg:.2f}."
                )
                p_bar.update()

        if args.local_rank in [-1, 0]:
            logger.info(
                f"{epoch+1}/{args.epochs}: lr:{scheduler.get_last_lr()[0]:.4f}, "
                f"loss:{losses.avg:.4f}, loss_x:{losses_x.avg:.4f}, "
                f"loss_u:{losses_u.avg:.4f}, mask:{mask_probs.avg:.2f}"
            )

        if not args.no_progress:
            p_bar.close()

        # ── Evaluation and checkpointing (rank 0 only) ────────────────────────
        test_model = ema_model.ema if args.use_ema else model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)
            is_best  = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            ema_to_save   = (ema_model.ema.module
                             if hasattr(ema_model.ema, "module") else ema_model.ema) \
                             if args.use_ema else None

            save_checkpoint({
                'epoch':          epoch + 1,
                'state_dict':     model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc':            test_acc,
                'best_acc':       best_acc,
                'optimizer':      optimizer.state_dict(),
                'scheduler':      scheduler.state_dict(),
                'time_p':         time_p,
                'p_model':        p_model,
                'label_hist':     label_hist,
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info(f'Best top-1 acc: {best_acc:.2f}')
            logger.info(f'Mean top-1 acc (last 20): {np.mean(test_accs[-20:]):.2f}\n')


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

def cosine_similarity_loss(logits1, logits2):
    """
    Computes the mean cosine similarity between two sets of logit vectors.

    Used as the inter-block alignment loss L_align: maximising this quantity
    encourages the two output blocks to produce consistent predictions on
    unlabelled data, reducing persistent disagreements when labelled data
    is scarce. It is subtracted from the total loss (not added) since it
    is a quantity to maximise.

    Args:
        logits1, logits2 : tensors of shape (B, num_classes)

    Returns:
        Scalar mean cosine similarity in [-1, 1].
    """
    logits1_norm = F.normalize(logits1, p=2, dim=-1)
    logits2_norm = F.normalize(logits2, p=2, dim=-1)
    cosine_sim   = (logits1_norm * logits2_norm).sum(dim=-1)  # dot product of unit vectors
    return cosine_sim.mean()


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def accuracy(args, outputs, targets, topk=(1,)):
    """
    Computes top-k accuracy for the sMLP multi-block output.

    Averages softmax probabilities across all n_blocks output blocks before
    taking the argmax. This soft voting at inference time is the standard
    evaluation protocol for the sMLP (cf. Eq. (test_inference) in the paper).

    Args:
        outputs : raw logits of shape (B, n_blocks * num_classes)
        targets : ground-truth labels of shape (B,)
        topk    : tuple of k values to evaluate (default: top-1 and top-5)

    Returns:
        List of top-k accuracy values (one per k).
    """
    # Average softmax probabilities across blocks
    proba = torch.zeros(outputs.size(0), args.num_classes).to(args.device)
    for i in range(args.n_blocks):
        output = outputs[:, i * args.num_classes:(i + 1) * args.num_classes]
        proba += torch.softmax(output, dim=1) / args.n_blocks

    maxk       = max(topk)
    batch_size = targets.size(0)
    _, pred    = proba.topk(maxk, dim=1, largest=True, sorted=True)
    pred       = pred.t()
    correct    = pred.eq(targets.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test(args, test_loader, model, epoch):
    """
    Evaluates the model on the full test set.

    Reports top-1 and top-5 accuracy using soft-voted sMLP predictions.
    Only called on rank 0 in distributed training.

    Returns:
        (mean test loss, top-1 accuracy)
    """
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()
    end        = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs  = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)

            # Average CE loss across blocks (for logging only)
            loss_blocks = sum(
                F.cross_entropy(outputs[:, i * args.num_classes:(i + 1) * args.num_classes],
                                targets, reduction='mean')
                for i in range(args.n_blocks)
            )
            loss = loss_blocks / args.n_blocks

            prec1, prec5 = accuracy(args, outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(),  inputs.shape[0])
            top5.update(prec5.item(),  inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                test_loader.set_description(
                    f"Test {batch_idx+1}/{len(test_loader)}. "
                    f"Loss: {losses.avg:.4f}. top1: {top1.avg:.2f}. top5: {top5.avg:.2f}."
                )

        if not args.no_progress:
            test_loader.close()

    logger.info(f"top-1 acc: {top1.avg:.2f}")
    logger.info(f"top-5 acc: {top5.avg:.2f}")
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
