import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from .randaugment_cutmix import RandAugment
from copy import deepcopy
from dataset.mix_1 import rand_bbox
import random

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Dataset normalisation statistics
# ─────────────────────────────────────────────
# Per-channel mean and std computed over the STL-10 training+unlabelled set.
# Values are divided by 255 to match the [0, 1] range of ToTensor().
stl10_mean = tuple([x / 255 for x in [112.4, 109.1, 98.6]])
stl10_std  = tuple([x / 255 for x in [68.4,  66.6,  68.5]])

normal_mean = (0.5, 0.5, 0.5)
normal_std  = (0.5, 0.5, 0.5)


# ─────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────

def get_stl10(args, root, include_lb_to_ulb=True):
    """
    Builds the three STL-10 datasets used in SSL training.

    STL-10 specifics:
        - 'train'    split: 5,000 labelled images (500 per class, 96×96)
        - 'unlabeled' split: 100,000 unlabelled images
        - 'test'     split: 8,000 test images

    Following the standard SSL protocol, we use 1,000 labelled images
    (100 per class) drawn from the 'train' split. The unlabelled pool
    consists of the full 'unlabeled' split (100,000 images).

    When include_lb_to_ulb=True (default), the full 'train' split is
    also appended to the unlabelled pool with strong augmentation.
    This is the standard practice for STL-10 SSL benchmarks, as it
    maximises the amount of unlabelled data available during training.

    Args:
        args              : training arguments
        root              : path to dataset directory
        include_lb_to_ulb : if True, append labelled split to unlabelled pool

    Returns:
        train_labeled_dataset   : 1,000-sample labelled subset, weak augmentation
        train_unlabeled_dataset : unlabelled pool with TransformFixMatch
        test_dataset            : 8,000-sample test set, no augmentation
    """
    # Weak augmentation for labelled samples: flip + crop only
    # Note: STL-10 images are 96×96 — crop size is 96 (not 32 as in CIFAR)
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=96, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])

    # No augmentation for the test set
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])

    # Load the labelled 'train' split to draw the labelled subset
    base_dataset     = datasets.STL10(root, split='train',    download=True)
    # Load the dedicated unlabelled split (100,000 images, no labels)
    unlabeled_dataset = datasets.STL10(root, split='unlabeled', download=True)

    # Draw labelled indices (balanced across 10 classes)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.labels)

    # Labelled dataset: fixed subset of the 'train' split
    train_labeled_dataset = STL10SSL(
        root, train_labeled_idxs, split='train',
        transform=transform_labeled
    )

    # Unlabelled dataset: full 'unlabeled' split with multi-view transform
    train_unlabeled_dataset = STL10SSL(
        root, indexs=None, split='unlabeled',
        transform=TransformFixMatch(mean=stl10_mean, std=stl10_std)
    )

    # Append the full 'train' split (strongly augmented) to the unlabelled pool.
    # This is standard for STL-10 SSL: the 5,000 labelled images are also
    # used as unlabelled data, maximising the unlabelled pool size.
    if include_lb_to_ulb:
        train_strong_dataset = STL10SSL(
            root, indexs=None, split='train',
            transform=TransformFixMatch(mean=stl10_mean, std=stl10_std)
        )
        train_unlabeled_dataset = train_unlabeled_dataset + train_strong_dataset

    test_dataset = datasets.STL10(
        root, split='test', transform=transform_val, download=True
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


# ─────────────────────────────────────────────
# Labelled / unlabelled split
# ─────────────────────────────────────────────

def x_u_split(args, labels):
    """
    Splits the STL-10 'train' split into labelled and unlabelled subsets.

    Identical logic to the CIFAR version, adapted for STL-10:
        - label_per_class = num_labeled // num_classes samples drawn per class
        - Seed is fixed to args.seed_labeled when args.is_EL=1 so that all
          ensemble members share the same labelled split
        - Label expansion tiles the labelled index array to fill one full epoch
          when num_labeled < batch_size (common with 100 labels total)

    Note: unlike the CIFAR split, the world_size factor is not included in the
    expansion formula here (single-node training assumption for STL-10).

    Args:
        args   : training arguments
        labels : numpy array of integer class labels for the 'train' split

    Returns:
        labeled_idx   : balanced labelled indices (possibly expanded)
        unlabeled_idx : all 'train' split indices (used as additional unlabelled data)
    """
    # Fix seed for labelled split reproducibility across ensemble members
    if args.is_EL:
        random.seed(args.seed_labeled)
        np.random.seed(args.seed_labeled)

    label_per_class = args.num_labeled // args.num_classes
    labels          = np.array(labels)
    labeled_idx     = []

    # All 'train' samples are treated as unlabelled (standard SSL convention)
    unlabeled_idx = np.array(range(len(labels)))

    # Draw label_per_class samples per class without replacement
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, replace=False)
        labeled_idx.extend(idx)

    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    # Expand labelled indices if too few to fill one epoch
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled
        )
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


# ─────────────────────────────────────────────
# Multi-view transform for unlabelled samples
# ─────────────────────────────────────────────

class TransformFixMatch(object):
    """
    Produces four augmented views from a single STL-10 image.

    Identical pipeline to the CIFAR version (TransformFixMatch in
    cifar_cutmix_w.py), adapted for 96×96 images:
        - Crop size: 96 (not 32)
        - Padding: int(32 * 0.125) = 4 pixels

    Returns:
        weak    : weakly augmented view (flip + crop + normalise)
        strong1 : weak → RandAugment → normalise
        strong2 : weak → RandAugment → normalise (independent draw)
        bbox_w  : CutMix bounding box for the weak view
        lam_w   : CutMix mixing coefficient for the weak view
    """

    def __init__(self, mean, std):
        # Weak augmentation: flip + crop (96×96 crop for STL-10)
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=96, padding=int(32 * 0.125), padding_mode='reflect')
        ])

        # Two independent strong augmentations via RandAugment
        self.strong1 = transforms.Compose([RandAugment(3, 5, flag_using_random_num=True)])
        self.strong2 = transforms.Compose([RandAugment(3, 5, flag_using_random_num=True)])

        # Final normalisation (shared across all views)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        # Apply weak augmentation first (shared base for strong views)
        weak    = self.weak(x)
        strong1 = self.strong1(deepcopy(weak))
        strong2 = self.strong2(deepcopy(weak))

        # Normalise all views
        weak    = self.normalize(weak)
        strong1 = self.normalize(strong1)
        strong2 = self.normalize(strong2)

        # Pre-compute CutMix parameters on the weak view
        bbox_w, lam_w = rand_bbox(weak.size())

        return weak, strong1, strong2, bbox_w, lam_w


# ─────────────────────────────────────────────
# STL-10 SSL dataset wrapper
# ─────────────────────────────────────────────

class STL10SSL(datasets.STL10):
    """
    STL-10 subset dataset for SSL.

    Wraps torchvision's STL10 and filters to a given index set.
    Handles both the labelled 'train' split (indexed) and the full
    'unlabeled' split (indexs=None).

    Key difference from CIFAR wrappers: STL-10 images are stored as
    (N, C, H, W) in torchvision, so self.data is transposed to
    (N, H, W, C) to match the PIL Image.fromarray() convention.
    """

    def __init__(self, root, indexs, split='train', transform=None,
                 target_transform=None, download=False):
        super().__init__(root, split=split, transform=transform,
                         target_transform=target_transform, download=download)
        if indexs is not None:
            self.data   = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

        # STL-10 data is stored as (N, C, H, W); convert to (N, H, W, C)
        # so that Image.fromarray() receives the expected HWC layout
        self.data = self.data.transpose([0, 2, 3, 1])

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
