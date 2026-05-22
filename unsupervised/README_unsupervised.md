# Unsupervised Learning on MNIST with sMLP

This folder contains the unsupervised variant of the sMLP classifier applied to MNIST.
No external labels are used during training — the model learns from internally generated
pseudo-labels (Hard-LWTA codes from a frozen model snapshot).
A small set of labelled prototypes (1, 3, or 10 per class) is used **only at inference time**
for nearest-neighbour classification via cosine similarity.

---

## Method Overview

The model consists of a convolutional encoder followed by a two-layer sparse MLP (sMLP) head:

```
Input (28×28)
    → Conv block 1  (Conv → BN → ReLU → MaxPool)
    → Conv block 2  (Conv → BN → ReLU → MaxPool)
    → Flatten  →  4608-dim feature vector
    → pc1: sparse linear (sparsity s1) + Soft-LWTA
    → pc2: sparse linear (sparsity s2) + per-block min-max normalisation
    → Output: L2-dim binary-like code (L2/k2 blocks of k2 values)
```

**Training** follows a pseudo-label consistency strategy:
- A frozen snapshot of the model generates binary pseudo-labels (Hard-LWTA) from a weakly augmented view.
- The live model is trained on a strongly augmented view to match those pseudo-labels via BCE loss.

**Inference** is prototype-based:
- Mean embeddings are computed over the prototype set for each class.
- Test samples are classified by nearest prototype under cosine similarity.

---

## Files

| File | Description |
|---|---|
| `mnist_ssl.py` | Main training and evaluation script |
| `mnist_dataset.py` | Dataset, augmentation pipelines, and prototype loaders |

---

## Requirements

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm wandb
```

MNIST will be loaded from `./data/`. Set `download=True` in `mnist_dataset.py` on first run.

---

## Usage

```bash
python mnist_ssl.py \
    --epochs 100 \
    --L1 1600 \
    --L2 1200 \
    --k1 8 \
    --k2 8 \
    --s1 0.85 \
    --s2 0.96 \
    --batch-size 64 \
    --lr-init 0.0015
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--L1` | 1600 | Output size of the first sparse layer (must be divisible by `k1`) |
| `--L2` | 1200 | Output size of the second sparse layer (must be divisible by `k2`) |
| `--k1` | 8 | Block size for Soft-LWTA in the intermediate layer |
| `--k2` | 8 | Block size for Hard-LWTA in the output layer |
| `--s1` | 0.85 | Sparsity rate of pc1 |
| `--s2` | 0.96 | Sparsity rate of pc2 |
| `--epochs` | 100 | Number of training epochs |
| `--n-iters` | 5 | Passes over the unlabelled loader per epoch |

---

## Prototype Sets

Three prototype sets are provided in `mnist_dataset.py`, using fixed indices from the MNIST training set:

| Set | Prototypes per class | Total |
|---|---|---|
| `indices_to_keep_1` | 1 | 10 |
| `indices_to_keep_3` | 3 | 30 |
| `indices_to_keep_10` | 10 | 100 |

By default, `proto_loader[0]` (1 prototype per class) is used for evaluation.

---

## Logging

Training metrics (accuracy, loss) are logged to [Weights & Biases](https://wandb.ai).
Set your project name in `mnist_ssl.py`:
```python
wandb.init(project='MNIST SSL', ...)
```

---

## Reference

> B. Oubaha, C. Berrou, X. Ji, Y. Nasser, R. Le Bidan,
> *"On Diversity in Discriminative Neural Networks"*,
> IEEE 12th International Symposium on Signal, Image, Video and Communications (ISIVC), 2024.
> DOI: [10.1109/ISIVC61350.2024.10577798](https://doi.org/10.1109/ISIVC61350.2024.10577798)
