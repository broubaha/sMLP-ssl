# Adapted from:
#   - ildoonet/pytorch-randaugment: https://github.com/ildoonet/pytorch-randaugment
#   - rpmcruz/autoaugment: https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# Modified for the FixMatch / InfoMatch SSL pipeline with Cutout appended after RandAugment.

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────
# Individual augmentation operations
# Each function takes (img: PIL.Image, v: float) and returns a PIL.Image.
# The parameter v is sampled uniformly from [min_val, max_val] defined
# in augment_list() for each operation.
# ─────────────────────────────────────────────

def AutoContrast(img, _):
    """Maximises image contrast by remapping the pixel range to [0, 255]."""
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    """Adjusts brightness by factor v. v=1.0 → original; v<1 → darker."""
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    """Adjusts colour saturation by factor v. v=0 → grayscale; v=1 → original."""
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    """Adjusts contrast by factor v. v=1.0 → original."""
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    """Equalises the image histogram to produce a uniform intensity distribution."""
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    """Inverts all pixel values (negative image)."""
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    """No-op: returns the image unchanged. Acts as a skip operation in the policy."""
    return img


def Posterize(img, v):
    """
    Reduces the number of bits per colour channel to v bits.
    v is clipped to [1, 8]; lower values produce more posterised (banded) images.
    """
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):
    """Rotates the image by v degrees. v in [-30, 30]."""
    return img.rotate(v)


def Sharpness(img, v):
    """Adjusts sharpness by factor v. v=1.0 → original; v=2.0 → sharper."""
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):
    """Applies horizontal shear by factor v. v in [-0.3, 0.3]."""
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    """Applies vertical shear by factor v. v in [-0.3, 0.3]."""
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):
    """
    Translates horizontally by v * image_width pixels.
    v in [-0.3, 0.3] (relative fraction of image width).
    """
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):
    """Translates horizontally by v pixels (absolute, not relative)."""
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    """
    Translates vertically by v * image_height pixels.
    v in [-0.3, 0.3] (relative fraction of image height).
    """
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):
    """Translates vertically by v pixels (absolute, not relative)."""
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):
    """
    Inverts all pixel values above threshold v.
    v=256 → no effect; v=0 → fully inverted.
    """
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):
    """
    Erases a random square patch of relative size v from the image.
    v in [0, 0.5] (fraction of image width). The patch is filled with
    a fixed grey colour (125, 123, 114) — the mean ImageNet colour.
    v=0 → no erasure.
    """
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img
    v = v * img.size[0]   # convert relative size to absolute pixels
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    """
    Erases a square patch of absolute size v×v pixels at a random location.
    The patch centre is drawn uniformly; corners are clipped to image boundaries.
    Filled with grey (125, 123, 114).
    """
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    # Top-left corner of the patch
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy    = (x0, y0, x1, y1)
    color = (125, 123, 114)   # approximate mean ImageNet RGB colour
    img   = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


# ─────────────────────────────────────────────
# Augmentation policy
# ─────────────────────────────────────────────

def augment_list():
    """
    Returns the pool of augmentation operations used by RandAugment.

    Each entry is a tuple (operation, min_val, max_val). The magnitude
    parameter v is sampled uniformly from [min_val, max_val] at call time.

    The pool follows the FixMatch / RandAugmentMC convention:
    14 operations covering photometric and geometric transformations.
    Cutout is applied separately after the main RandAugment loop.
    """
    l = [
        (AutoContrast, 0,     1   ),
        (Brightness,   0.05,  0.95),
        (Color,        0.05,  0.95),
        (Contrast,     0.05,  0.95),
        (Equalize,     0,     1   ),
        (Identity,     0,     1   ),   # skip op — preserves augmentation diversity
        (Posterize,    4,     8   ),
        (Rotate,       -30,   30  ),
        (Sharpness,    0.05,  0.95),
        (ShearX,       -0.3,  0.3 ),
        (ShearY,       -0.3,  0.3 ),
        (Solarize,     0,     256 ),
        (TranslateX,   -0.3,  0.3 ),
        (TranslateY,   -0.3,  0.3 ),
    ]
    return l


# ─────────────────────────────────────────────
# RandAugment
# ─────────────────────────────────────────────

class RandAugment:
    """
    RandAugment strong augmentation policy for SSL.

    Randomly selects n operations from the augmentation pool and applies
    them sequentially with uniformly sampled magnitudes. A random Cutout
    is always appended after the n operations.

    This follows the FixMatch convention for strong augmentation:
        strong = weak_view → RandAugment(n=3) → Cutout

    Args:
        n                    : number of operations to sample and apply
        m                    : legacy magnitude parameter (not used — magnitude
                               is sampled uniformly from each op's range instead)
        flag_using_random_num: if True, the number of ops would be randomised
                               (currently unused — kept for future ablations)

    Usage:
        augmentor = RandAugment(n=3, m=5, flag_using_random_num=True)
        augmented_img = augmentor(pil_image)
    """

    def __init__(self, n, m, flag_using_random_num=False):
        self.n                    = n
        self.m                    = m          # deprecated; magnitude sampled per-op instead
        self.augment_list         = augment_list()
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        # Sample n operations uniformly at random (with replacement)
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # Sample magnitude uniformly from the operation's valid range
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)

        # Always apply Cutout after the main operations (FixMatch convention)
        # Cutout size is sampled uniformly in [0, 0.5] relative to image width
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)

        return img


# ─────────────────────────────────────────────
# Quick visual test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os
    import matplotlib
    from matplotlib import pyplot as plt

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    img     = PIL.Image.open('./u.jpg')
    randaug = RandAugment(3, 6)
    img     = randaug(img)

    plt.imshow(img)
    plt.show()
