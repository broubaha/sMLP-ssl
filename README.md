# sMLP-ssl: Sparse MLP for Unsupervised and Semi-Supervised Learning

Official code for the paper:

> **"The Value of Sparse Connections in Neural Networks for Weakly Supervised Learning"**
> Brahim Oubaha, Claude Berrou, Yehya Nasser, Raphaël Le Bidan
> *IEEE Transactions on Neural Networks and Learning Systems*, 2025 (under review)

---

## Overview

This repository contains two independent sets of experiments:

| Folder | Task | Datasets |
|---|---|---|
| `unsupervised/` | Fully unsupervised classification | MNIST |
| `weakly_supervised/` | Semi-supervised classification | CIFAR-10, CIFAR-100, STL-10 |

Both share the same core idea: replacing the standard dense one-hot classification head with a **sparse MLP (sMLP)** decoder combining three principles:

- **Sparse connections** — randomly drawn, fixed throughout training
- **Local Winner-Take-All (LWTA)** competition — sample-specific activation patterns
- **Multi-block output** — repetition coding for intra- and extra-diversity

---

## Repository Structure

```
sMLP-ssl/
├── README.md
│
├── unsupervised/               ← MNIST fully unsupervised
│   ├── mnist_ssl.py            # Training and evaluation
│   ├── mnist_dataset.py        # Dataset, augmentations, prototype loaders
│   └── README.md
│
└── weakly_supervised/          ← CIFAR-10/100 and STL-10 semi-supervised
    ├── train_cifar_stl.py      # Main training script (DDP, HPC)
    ├── dataset/
    │   ├── cifar_cutmix_w.py   # CIFAR-10/100 datasets and multi-view transforms
    │   ├── STL10.py            # STL-10 dataset and multi-view transforms
    │   ├── mix_1.py            # MixUp and CutMix augmentation functions
    │   └── randaugment_cutmix.py # RandAugment strong augmentation
    ├── models/
    │   ├── wideresnet_e.py     # WideResNet encoder + sMLP head
    │   └── ema_wrn.py          # Exponential Moving Average model wrapper
    └── utils/
        └── misc.py             # AverageMeter utility
```

---

## The sMLP Architecture

The sMLP replaces the standard dense classification head:

```
Encoder output (l dims)
    → pc1 : sparse linear (sparsity s1) + BN + Soft-LWTA
    → pc2 : sparse linear (sparsity s2)
    → Output : B2 blocks of C logits  (B2 × C total)
```

At inference, softmax probabilities are averaged across all B2 blocks:

$$P(y) = \frac{1}{B_2} \sum_{b=1}^{B_2} P_b(y)$$

---

## Results

### Semi-Supervised Learning — Error rates (%) ↓

| Method | CIFAR-10 40 | CIFAR-10 250 | CIFAR-10 4000 | CIFAR-100 400 | CIFAR-100 2500 | STL-10 1000 |
|---|---|---|---|---|---|---|
| InfoMatch (reprod.) | 6.32 | 4.33 | 3.73 | 38.68 | 23.73 | 6.17 |
| **sMLP (ours)** | **4.82** | **4.18** | **3.68** | 38.35 | **23.70** | **4.56** |

### Ensemble (4 models) — Error rates (%) ↓

| Method | CIFAR-10 40 | CIFAR-10 250 | CIFAR-10 4000 | CIFAR-100 400 | CIFAR-100 2500 | STL-10 1000 |
|---|---|---|---|---|---|---|
| InfoMatch (reprod.) | 3.99 | 3.88 | 3.30 | 37.29 | 22.90 | 5.61 |
| **sMLP (ours)** | **3.74** | **3.37** | **3.14** | **34.31** | **22.65** | **3.95** |

---

## Requirements

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm wandb
```

> **Note on distributed training**: `weakly_supervised/train_cifar_stl.py` uses
> [`idr_torch`](http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-torch-multi.html),
> a utility available on the Jean Zay and Odyssey HPC clusters (IDRIS/IMT Atlantique).
> To run on a different cluster or locally, replace `idr_torch` calls with standard
> `torch.distributed` initialisation.

---

## Usage

### Unsupervised (MNIST)

```bash
cd unsupervised
python mnist_ssl.py --epochs 100 --L1 1600 --L2 1200 --k1 8 --k2 8 --s1 0.85 --s2 0.96
```

See [`unsupervised/README.md`](unsupervised/README.md) for full details.

### Semi-Supervised (CIFAR-10/100, STL-10)

```bash
cd weakly_supervised

# sMLP on CIFAR-100, 400 labels
python train_cifar_stl.py \
    --dataset cifar100 \
    --num-labeled 400 \
    --arch wideresnet \
    --n-blocks 2 \
    --s1 0.92 --s2 0.92 \
    --l1 255 --k1 3 \
    --lam-sim 0.2 \
    --seed 1

# InfoMatch baseline
python train_cifar_stl.py \
    --dataset cifar100 \
    --num-labeled 400 \
    --baseline \
    --seed 1
```

See [`weakly_supervised/`](weakly_supervised/) for full argument descriptions.

---

## Citation

If you use this code, please cite:

```bibtex
@article{oubaha2025smlp,
  title   = {The Value of Sparse Connections in Neural Networks for Weakly Supervised Learning},
  author  = {Oubaha, Brahim and Berrou, Claude and Nasser, Yehya and Le Bidan, Rapha{\"e}l},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2025},
  note    = {Under review}
}
```

---

## Related Publications

This work extends our earlier study on diversity in discriminative neural networks:

```bibtex
@inproceedings{oubaha2024diversity,
  title     = {On Diversity in Discriminative Neural Networks},
  author    = {Oubaha, Brahim and Berrou, Claude and Ji, Xueyao and Nasser, Yehya and Le Bidan, Rapha{\"e}l},
  booktitle = {Proceedings of the IEEE 12th International Symposium on Signal, Image, Video and Communications (ISIVC)},
  pages     = {1--6},
  year      = {2024},
  doi       = {10.1109/ISIVC61350.2024.10577798}
}
```

---

## Acknowledgements

This work was performed using HPC resources from GENCI-IDRIS (Grant 2024-AD011015936).
The semi-supervised training pipeline builds on
[InfoMatch](https://github.com/kekmodel/FixMatch-pytorch) and the
[FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) codebase.
