import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import random
from .randaugment_cutmix import RandAugment
from copy import deepcopy
from dataset.mix_1 import rand_bbox
from .STL10 import get_stl10

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Dataset normalisation statistics
# ─────────────────────────────────────────────
# Per-channel mean and std computed over the full training sets.
# Used to normalise images to approximately zero mean and unit variance.
cifar10_mean  = (0.4914, 0.4822, 0.4465)
cifar10_std   = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std  = (0.2675, 0.2565, 0.2761)
normal_mean   = (0.5, 0.5, 0.5)
normal_std    = (0.5, 0.5, 0.5)


# ─────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────

def get_cifar10(args, root, index=None):
    """
    Builds the three CIFAR-10 datasets used in SSL training.

    Returns:
        train_labeled_dataset   : labelled subset with weak augmentation
        train_unlabeled_dataset : full training set with TransformFixMatch
                                  (weak + two strong views + CutMix params)
        test_dataset            : standard test set, no augmentation
    """
    # Weak augmentation for labelled samples: only flip + crop
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    # No augmentation for the test set
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    # Split training indices into labelled and unlabelled subsets
    if index is not None:
        # Use a fixed index set (e.g. for the 10-label FixMatch protocol)
        train_labeled_idxs   = index
        train_unlabeled_idxs = np.array(range(len(base_dataset.targets)))
    else:
        train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )
    # Unlabelled samples receive the full multi-view transform (weak + strong × 2 + CutMix)
    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
    )
    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):
    """
    Builds the three CIFAR-100 datasets used in SSL training.

    Same structure as get_cifar10, with CIFAR-100 specific statistics
    and the wider class space (100 classes).

    Returns:
        train_labeled_dataset, train_unlabeled_dataset, test_dataset
    """
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    base_dataset = datasets.CIFAR100(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )
    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std)
    )
    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


# ─────────────────────────────────────────────
# Labelled / unlabelled split
# ─────────────────────────────────────────────

def x_u_split(args, labels):
    """
    Splits the training set into a small labelled subset and a large
    unlabelled subset, following the standard SSL protocol.

    Labelled split:
        - Exactly args.num_labeled samples total, balanced across classes
          (label_per_class = num_labeled // num_classes samples per class).
        - Drawn with a fixed seed (args.seed_labeled) when args.is_EL=1,
          so that all ensemble members share the same labelled split.

    Unlabelled split:
        - All training samples (including those in the labelled split),
          following the FixMatch convention.

    Label expansion:
        - If num_labeled < batch_size * world_size, the labelled index array
          is tiled to fill at least one full epoch of eval_step iterations.
          This avoids running out of labelled samples mid-epoch.

    Args:
        args   : training arguments (num_labeled, num_classes, seed_labeled, ...)
        labels : list of integer class labels for the full training set

    Returns:
        labeled_idx   : numpy array of labelled sample indices (possibly expanded)
        unlabeled_idx : numpy array of all training indices
    """
    # Fix seed for labelled split only (ensemble members share the same labels)
    if args.is_EL:
        random.seed(args.seed_labeled)
        np.random.seed(args.seed_labeled)

    label_per_class = args.num_labeled // args.num_classes
    labels          = np.array(labels)
    labeled_idx     = []

    # All training samples are used as unlabelled data (standard SSL protocol)
    unlabeled_idx = np.array(range(len(labels)))

    # Draw label_per_class random indices per class without replacement
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, replace=False)
        labeled_idx.extend(idx)

    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    # Expand labelled indices if the labelled set is too small for one epoch
    if args.expand_labels or args.num_labeled < args.batch_size * args.world_size:
        num_expand_x = math.ceil(
            (args.batch_size * args.world_size) * args.eval_step / args.num_labeled
        )
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


# ─────────────────────────────────────────────
# Multi-view transform for unlabelled samples
# ─────────────────────────────────────────────

class TransformFixMatch(object):
    """
    Produces four augmented views from a single image, following the
    FixMatch / InfoMatch protocol extended with CutMix:

        weak    : horizontal flip + random crop (used for pseudo-label generation)
        strong1 : weak → RandAugment (used for consistency loss, view 1)
        strong2 : weak → RandAugment (used for consistency loss, view 2)
        bbox_w  : CutMix bounding box computed on the weak view
        lam_w   : CutMix mixing coefficient for the weak view

    The two strong views are independent draws of RandAugment applied on
    top of the same weak view. This ensures that pseudo-labels (from the
    weak view) are more reliable than the strongly augmented predictions.

    The CutMix parameters (bbox_w, lam_w) are pre-computed here and passed
    to the training loop, where cutmix_hard() applies them to mix pairs of
    weak unlabelled views on-the-fly.
    """

    def __init__(self, mean, std):
        # Weak augmentation: flip + crop only (no colour distortion)
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect')
        ])

        # Two independent strong augmentations via RandAugment
        # flag_using_random_num=True: number of ops is also randomised per call
        self.strong1 = transforms.Compose([RandAugment(3, 5, flag_using_random_num=True)])
        self.strong2 = transforms.Compose([RandAugment(3, 5, flag_using_random_num=True)])

        # Final normalisation applied to all views
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        # Apply weak augmentation first (shared base for all views)
        weak    = self.weak(x)

        # Strong views are built on top of the weak view
        strong1 = self.strong1(deepcopy(weak))
        strong2 = self.strong2(deepcopy(weak))

        # Normalise all views to tensors
        weak    = self.normalize(weak)
        strong1 = self.normalize(strong1)
        strong2 = self.normalize(strong2)

        # Pre-compute CutMix box and coefficient for the weak view
        # These will be used in train() to mix pairs of unlabelled weak views
        bbox_w, lam_w = rand_bbox(weak.size())

        return weak, strong1, strong2, bbox_w, lam_w


# ─────────────────────────────────────────────
# Utility: standalone strong augmentation
# ─────────────────────────────────────────────

# Pre-instantiated transform for extracting a single strong view.
# Used outside the main training loop when only one augmented view is needed.
transform_fixmatch = TransformFixMatch(cifar10_mean, cifar10_std)

def strong_augmentation(x):
    """Returns a single strongly augmented view of image x (CIFAR-10 stats)."""
    _, strong1, _, _, _ = transform_fixmatch(x)
    return strong1


# ─────────────────────────────────────────────
# SSL dataset wrappers
# ─────────────────────────────────────────────

class CIFAR10SSL(datasets.CIFAR10):
    """
    CIFAR-10 subset dataset for SSL.

    Wraps torchvision's CIFAR10 and filters samples to a given index set.
    Used to create both the labelled subset (small, fixed indices) and the
    unlabelled subset (all training indices).
    """
    def __init__(self, root, indexs, train=True, transform=None,
                 target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        if indexs is not None:
            self.data    = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    """
    CIFAR-100 subset dataset for SSL.

    Identical structure to CIFAR10SSL; inherits from CIFAR100 instead.
    """
    def __init__(self, root, indexs, train=True, transform=None,
                 target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        if indexs is not None:
            self.data    = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

# Maps dataset name strings (used in argparse) to their builder functions.
# STL-10 is handled by a separate module (dataset/STL10.py).
DATASET_GETTERS = {
    'cifar10':  get_cifar10,
    'cifar100': get_cifar100,
    'STL10':    get_stl10
}
