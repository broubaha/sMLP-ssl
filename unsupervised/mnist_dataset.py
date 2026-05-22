import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Multi-view transform wrappers
# ─────────────────────────────────────────────

class TwoTransform:
    """
    Applies two different transforms to the same image and returns both.
    Used to generate (weak, strong) augmentation pairs for consistency training.
    """
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]


class threetransform:
    """
    Applies three transforms to the same image and returns all three views.

    Returns:
        [original, weak_aug, strong_aug]

    Used by the training loader to provide:
        - a reference view (trans)
        - a weakly augmented view for pseudo-label generation (trans_w)
        - a strongly augmented view for consistency loss (trans_strong)
    """
    def __init__(self, trans, trans_w, trans_strong):
        self.trans        = trans
        self.trans_w      = trans_w
        self.trans_strong = trans_strong

    def __call__(self, x):
        img1 = self.trans(x)
        img2 = self.trans_w(x)
        img3 = self.trans_strong(x)
        return [img1, img2, img3]


# ─────────────────────────────────────────────
# Individual augmentation functions
# Applied as part of the strong augmentation pipeline.
# Each operates on a batch tensor of shape (B, 1, 28, 28).
# ─────────────────────────────────────────────

def single_rotate(data):
    """Random rotation in [-20°, +20°] applied independently per sample."""
    data1 = data.clone().reshape(-1, 1, 28, 28)
    for idx in range(data1.shape[0]):
        data1[idx] = torchvision.transforms.functional.rotate(
            data1[idx], angle=40 * float(torch.rand(1)) - 20.0
        )
    return data1


def single_elasticity(data, ds=0.4):
    """
    Random perspective distortion applied independently per sample.
    distortion_scale=0.4 controls the strength of the geometric warp.
    """
    data1 = data.clone().reshape(-1, 1, 28, 28)
    for idx in range(data1.shape[0]):
        perspective_transformer = torchvision.transforms.RandomPerspective(ds, p=float(torch.rand(1)))
        data1[idx] = perspective_transformer(data1[idx])
    return data1


def single_erasing(data, p=0.5, w=5):
    """
    Random erasing of a square patch of size w×w, applied with probability p.
    Erased region is filled with zeros (black).
    """
    data1 = data.clone().reshape(-1, 1, 28, 28)
    for idx in range(data1.shape[0]):
        eraser = torchvision.transforms.RandomErasing(
            p=p,
            scale=([(w * w) / (28 * 28), (w * w) / (28 * 28)]),
            ratio=([1, 1]),
            value=0,
            inplace=False
        )
        data1[idx] = eraser(data1[idx])
    return data1


def single_centercrop(data, p=0.5, s=20):
    """
    Center-crops to size s×s and resizes back to 28×28, applied with probability p.
    Simulates zoom-in augmentation.
    """
    data1 = data.clone().reshape(-1, 1, 28, 28)
    centercrop = torchvision.transforms.CenterCrop(size=s)
    resize     = torchvision.transforms.Resize(data1.shape[-1], antialias=True)
    for idx in range(data1.shape[0]):
        if torch.rand(1) > p:
            data1[idx] = resize(centercrop(data1[idx]))
    return data1


def single_ColorJitter(data1, p=0.5, b=0.5, h=0.3):
    """
    Random brightness and hue jitter applied with probability p.
    Note: operates on grayscale MNIST, so hue has no effect in practice.
    """
    data1 = data.clone().reshape(-1, 1, 28, 28)
    for idx in range(data1.shape[0]):
        if torch.rand(1) > p:
            color_jitter = torchvision.transforms.ColorJitter(brightness=b, hue=h)
            data1[idx] = color_jitter(data1[idx])
    return data1


# ─────────────────────────────────────────────
# Transform pipelines
# ─────────────────────────────────────────────

# Strong augmentation: normalisation + successive geometric/occlusion transforms.
# Applied to unlabelled samples during training (consistency regularisation).
transform_strong = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1, 1, 28, 28)),
    single_centercrop,   # random zoom-in
    single_erasing,      # random patch occlusion
    single_elasticity,   # random perspective warp
    single_rotate        # random rotation
])

# Standard transform: normalisation only.
# Applied to labelled prototypes and test samples.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1, 1, 28, 28))
])


# ─────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────

def get_data_loader(batch_size=64, seed=42, keep=[]):
    """
    Builds and returns the three data loaders used in training:

        trainset    : full MNIST training set with three-view augmentation
                      (original, weak, strong) — used for pseudo-label training
        proto_loader: list of three DataLoaders for prototype sets of sizes
                      1, 3, and 10 samples per class respectively.
                      Prototypes are used at inference time for cosine-similarity
                      based nearest-neighbour classification.
        test_loader : standard MNIST test set (10,000 samples, no augmentation)

    Prototype indices are fixed across runs to ensure reproducible evaluation.
    Three prototype sets are provided to allow experiments with different
    numbers of labelled references per class (1, 3, or 10).

    Args:
        batch_size : mini-batch size for train and test loaders
        seed       : random seed for full reproducibility
        keep       : unused (kept for API compatibility)

    Returns:
        trainset, proto_loader (list of 3 loaders), test_loader
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Training set: three-view augmentation (original, weak, strong)
    trainset = datasets.MNIST(
        root='./data', train=True, download=False,
        transform=threetransform(transform, transform_strong, transform_strong)
    )

    # Test set: no augmentation
    testset = datasets.MNIST(
        root='./data', train=False, download=False,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # --- Prototype index sets ---
    # Each list contains exactly 1 index per class (10 total).
    # Multiple candidate sets were tested; the last assignment is the one used.
    indices_to_keep_1 = [38501, 50225, 55749, 10775, 9566, 43587, 7206, 41275, 57640, 23217]

    # 3 prototypes per class (30 total)
    indices_to_keep_3 = [
        320, 269, 365, 50, 338, 6772, 430, 560, 708, 116,
        21, 24, 16, 30, 9, 236, 32, 38, 240, 48,
        260, 70, 25, 452, 92, 266, 93, 158, 312, 54
    ]

    # 10 prototypes per class (100 total)
    indices_to_keep_10 = [
        47756, 27620, 38232, 36667, 4023, 39531, 19012, 6422, 30187, 40820,
        24924, 40769, 1155, 42371, 26009, 38039, 49205, 4998, 59207, 58171,
        18065, 10941, 49285, 40107, 33593, 42219, 21443, 477, 52016, 23677,
        857, 45264, 32476, 50214, 30306, 767, 13142, 8821, 10760, 37730,
        55323, 16995, 23375, 3280, 13765, 48013, 8634, 55773, 45538, 12384,
        53352, 49788, 20718, 45124, 4845, 46829, 38759, 29845, 33200, 20017,
        37451, 34993, 8145, 32140, 12520, 57490, 51476, 27624, 38394, 51232,
        19312, 22513, 18002, 5825, 7616, 1491, 42831, 44562, 3507, 10243,
        33193, 51841, 17564, 3865, 31355, 36192, 12616, 25583, 30654, 14690,
        38899, 7890, 57753, 28197, 35188, 16476, 15584, 47745, 52845, 21571
    ]

    # Prototype dataset: no augmentation (clean reference embeddings)
    trainset_proto = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=transform
    )

    # Build one DataLoader per prototype set size
    keep = [indices_to_keep_1, indices_to_keep_3, indices_to_keep_10]
    proto_loader = [
        torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset_proto, indices_to_keep),
            batch_size=1, shuffle=True
        )
        for indices_to_keep in keep
    ]

    return trainset, proto_loader, test_loader


# ─────────────────────────────────────────────
# Visualisation utility
# ─────────────────────────────────────────────

def show_and_save_prototypes(proto_loader, num_images=10, save_path="prototypes.png"):
    """
    Displays and saves the first num_images samples from a prototype DataLoader.
    Useful for visually verifying that the selected prototype indices are clean
    and representative of their respective classes.

    Args:
        proto_loader : DataLoader yielding (image, label) pairs
        num_images   : number of images to display
        save_path    : file path to save the figure
    """
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i, (img, label) in enumerate(proto_loader):
        if i >= num_images:
            break
        img = img.squeeze().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Label: {label.item()}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {save_path}")