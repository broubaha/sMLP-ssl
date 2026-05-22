import argparse
import os
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvT_F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import wandb

from WideResnet import WarmupCosineLrScheduler
from mnist_dataset import get_data_loader

# ─────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────
torch.set_printoptions(profile="full")
plt.switch_backend('agg')  # Non-interactive backend (no display required, safe for HPC)

wandb.init(project='MNIST SSL', config={"dataset": "MNIST"}, save_code=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device is", device)

# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SSL model for MNIST")

# --- Data ---
parser.add_argument("--batch-size",    type=int,   default=64)
parser.add_argument("--c",             type=int,   default=1,      help="number of input channels (1 for grayscale MNIST)")
parser.add_argument("--w",             type=int,   default=28,     help="input image width")
parser.add_argument("--h",             type=int,   default=28,     help="input image height")
parser.add_argument("--dataset",       type=str,   default="mnist")
parser.add_argument("--dataset-path",  type=str,   default="./data/")
parser.add_argument("--dataset-size",  type=int,   default=18000)
parser.add_argument("--split",         type=float, default=1.0,    help="fraction of training samples to use per epoch")
parser.add_argument("--split-test",    type=float, default=1.0,    help="fraction of test samples to use")

# --- Optimisation ---
parser.add_argument("--lr-init",  type=float, default=0.0015,  help="initial learning rate")
parser.add_argument("--lr-end",   type=float, default=0.0001,  help="final learning rate (not used directly, kept for reference)")
parser.add_argument("--epochs",   type=int,   default=100)
parser.add_argument("--n-iters",  type=int,   default=5,        help="number of passes over the unlabelled loader per epoch")

# --- sMLP Architecture ---
# The classifier head is a two-layer sparse MLP (sMLP):
#   - pc1: sparse linear layer with Soft-LWTA activation  → intermediate representation
#   - pc2: sparse linear layer with Hard-LWTA activation  → binary multi-block output code
parser.add_argument("--L1",       type=int,   default=1600, help="output size of the first sparse layer (pc1), must be divisible by k1")
parser.add_argument("--L2",       type=int,   default=1200, help="output size of the second sparse layer (pc2), must be divisible by k2")
parser.add_argument("--k1",       type=int,   default=8,    help="block size for Soft-LWTA in pc1 (competition window)")
parser.add_argument("--k2",       type=int,   default=8,    help="block size for Hard-LWTA in pc2 (number of classes per block)")
parser.add_argument("--s1",       type=float, default=0.85, help="sparsity rate for pc1 (fraction of weights set to zero)")
parser.add_argument("--s2",       type=float, default=0.96, help="sparsity rate for pc2 (fraction of weights set to zero)")

# --- Device ---
parser.add_argument("--device",         type=str, default="cuda:0")
parser.add_argument("--dataset-device", type=str, default="")

# Support both script and notebook execution
try:
    get_ipython()
    args = parser.parse_args(args=[])   # Jupyter: use defaults
except NameError:
    args = parser.parse_args()          # Script: parse CLI arguments

# ─────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────

def Soft_LWTA(x, k1=None):
    """
    Soft Local Winner-Take-All (Soft-LWTA) activation.

    Divides the input into non-overlapping blocks of size k1.
    Within each block, the maximum value is kept and sub-maximum
    values are set to zero. The winning neuron retains its actual
    pre-activation value (soft version, as opposed to binary Hard-LWTA).

    Args:
        x  : input tensor of shape (batch_size, L1)
        k1 : block size (defaults to args.k1)

    Returns:
        Tensor of same shape as x with losers zeroed out per block.
    """
    if k1 is None:
        k1 = args.k1
    view = x.view(-1, x.shape[1] // k1, k1)        # reshape into blocks: (B, n_blocks, k1)
    max_vals, _ = view.max(dim=2, keepdim=True)     # find winner in each block
    view = view * (view >= max_vals)                # zero out all non-winners
    return view.view_as(x)                          # restore original shape


def Hard_LWTA(x, k2=None):
    """
    Hard Local Winner-Take-All (Hard-LWTA) activation.

    Divides the input into non-overlapping blocks of size k2.
    Within each block, only the winning neuron is set to 1;
    all others are set to 0. This produces a binary multi-block
    output code (one-hot per block), used as pseudo-labels.

    Args:
        x  : input tensor of shape (batch_size, L2)
        k2 : block size (defaults to args.k2)

    Returns:
        Binary tensor of same shape as x (one 1 per block, rest 0).
    """
    if k2 is None:
        k2 = args.k2
    view = x.reshape(-1, x.shape[1] // k2, k2).to(device)  # (B, n_blocks, k2)
    _, topk_indices = torch.topk(view, 1, dim=-1)           # index of winner in each block
    label = torch.zeros_like(view)
    label.scatter_(2, topk_indices, 1)                      # one-hot encoding per block
    return label.reshape(x.shape[0], -1)                    # flatten back to (B, L2)


# ─────────────────────────────────────────────
# Normalisation helper
# ─────────────────────────────────────────────

def min_max_normalize(output, epsilon=1e-4, k2=None):
    """
    Per-block min-max normalisation.

    Rescales each block of size k2 independently to [0, 1].
    Applied to the raw output of pc2 before loss computation or
    similarity-based inference. This ensures that cosine similarity
    and BCE loss operate on comparable scales across blocks.

    Args:
        output  : tensor of shape (batch_size, L2)
        epsilon : small constant for numerical stability
        k2      : block size (defaults to args.k2)

    Returns:
        Normalised tensor of same shape, values in [0, 1] per block.
    """
    if k2 is None:
        k2 = args.k2
    PC = output.reshape(output.shape[0], output.shape[1] // k2, k2).to(device)  # (B, n_blocks, k2)
    min_val = PC.min(dim=-1, keepdim=True)[0]
    max_val = PC.max(dim=-1, keepdim=True)[0]
    PC = (PC - min_val) / (max_val - min_val + epsilon)     # normalise each block
    return PC.reshape(output.shape[0], output.shape[1]).to(dtype=torch.float32)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class CSSLNet(nn.Module):
    """
    Convolutional Sparse SSL Network (CSSLNet).

    Architecture:
        Encoder  : two convolutional blocks (Conv → BN → ReLU → MaxPool)
        Decoder  : two-layer sMLP head
                   - pc1: sparse linear + BN + Soft-LWTA
                   - pc2: sparse linear + min-max normalisation per block

    The sMLP head replaces the standard dense one-hot classification layer.
    Sparsity is enforced via unstructured random pruning (PyTorch prune API),
    which zeros out a fraction of weights while keeping the mask fixed
    throughout training (no pruning schedule).

    Output: normalised continuous scores of shape (batch_size, L2),
            organised as L2/k2 blocks of k2 values each.
    """

    def __init__(self):
        super().__init__()

        # --- Encoder: convolutional backbone ---
        # Block 1: 1 → 64 channels, 5×5 conv, stride-1 max-pool
        self.conv1    = nn.Conv2d(1,  64,  kernel_size=5)
        self.bn1      = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Block 2: 64 → 128 channels, 5×5 conv, stride-3 max-pool
        self.conv2    = nn.Conv2d(64, 128, kernel_size=5)
        self.bn2      = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.relu = nn.ReLU(inplace=True)

        # --- Decoder: sparse MLP (sMLP) head ---
        # pc1: 4608 (encoder output) → L1, intermediate sparse layer
        self.pc1 = nn.Linear(4608,     args.L1)
        self.bn3 = nn.BatchNorm1d(args.L1)  # BN applied before Soft-LWTA competition

        # pc2: L1 → L2, output sparse layer producing L2/k2 blocks of k2 scores
        self.pc2 = nn.Linear(args.L1,  args.L2)

        # --- Weight initialisation ---
        # Small uniform init to avoid premature saturation before pruning
        nn.init.uniform_(self.pc1.weight, -0.001,  0.001)
        nn.init.uniform_(self.pc2.weight, -0.01,   0.01)

        # --- Sparse connectivity ---
        # Random unstructured pruning: zeros out a fraction s1 (resp. s2) of weights.
        # The pruning mask is fixed after initialisation (structural sparsity).
        prune.random_unstructured(self.pc1, name="weight", amount=args.s1)
        prune.random_unstructured(self.pc2, name="weight", amount=args.s2)

    def forward(self, x):
        # Encoder
        x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))   # (B, 64, H', W')
        x = self.maxpool2(self.relu(self.bn2(self.conv2(x))))   # (B, 128, H'', W'')
        x = x.reshape(x.size(0), -1)                           # flatten → (B, 4608)

        # sMLP decoder
        # Note: bn3 is kept in the model but currently bypassed (see commented line).
        # Soft-LWTA acts directly on pc1 raw outputs.
        #x = self.bn3(self.pc1(x))
        x = self.pc1(x)         # sparse linear: (B, L1)
        x = Soft_LWTA(x)        # winner-take-all per block: (B, L1)
        x = self.pc2(x)         # sparse linear: (B, L2)
        x = min_max_normalize(x) # per-block normalisation to [0, 1]: (B, L2)
        return x


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def init_weights(m):
    """
    Custom weight initialisation for convolutional layers.
    Applied via model.apply(init_weights) before training.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=0.5)


def calculate_sparsity_rates(model):
    """
    Computes the effective sparsity of pc1 and pc2 weight matrices.
    Sparsity = fraction of zero weights (after pruning mask is applied).
    """
    def sparsity(w):
        return 1 - torch.count_nonzero(w).item() / w.numel()
    return sparsity(model.pc1.weight.data), sparsity(model.pc2.weight.data)


def save_checkpoint(model, optimizer, scheduler, epoch, file_path):
    """Saves model, optimizer and scheduler states to disk."""
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, file_path)


# ─────────────────────────────────────────────
# Prototype assignment
# ─────────────────────────────────────────────

def assign(model, prototypes):
    """
    Computes model outputs for all prototype samples (one or a few
    labelled examples per class).

    Returns two dictionaries indexed by class label:
        assign_cluster_proba : continuous normalised scores (used for cosine similarity)
        assign_cluster_binai : binary Hard-LWTA codes (used for Hamming distance)
    """
    assign_cluster_proba = {i: [] for i in range(10)}
    assign_cluster_binai = {i: [] for i in range(10)}
    model.eval()
    with torch.no_grad():
        for data, label in prototypes:
            data  = data.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)
            output_scores = model(data.reshape(-1, args.c, args.w, args.h))
            estimated_label = Hard_LWTA(output_scores)
            assign_cluster_proba[int(label)].append(output_scores)
            assign_cluster_binai[int(label)].append(estimated_label)
    return assign_cluster_proba, assign_cluster_binai


def hamming_dist(assign_cluster_binai, epoch):
    """
    Computes the minimum pairwise Hamming distance between prototype
    binary codes across all class pairs. Used as a diversity indicator:
    a higher minimum distance means better class separation in the
    binary code space.
    """
    min_dist = 1000
    for key in assign_cluster_binai:
        for i in range(1, len(assign_cluster_binai)):
            if key + i in assign_cluster_binai:
                hdist = torch.cdist(
                    assign_cluster_binai[key][0],
                    assign_cluster_binai[key + i][0],
                    p=0                              # p=0: Hamming distance
                ).to(dtype=torch.long).squeeze(1)
                if hdist < min_dist:
                    min_dist = hdist
    return min_dist


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, epoch):
    """
    Runs one epoch of unsupervised pseudo-label training.

    Pseudo-labels are generated from a frozen snapshot of the model
    (model_backup) applied to weakly augmented images (ims_u). The
    main model is then trained on strongly augmented images (ims_u_strong)
    to match these pseudo-labels via BCE loss.

    This follows a FixMatch-style consistency regularisation strategy:
        weak view  → pseudo-label (no gradient)
        strong view → prediction  (gradient flows here)

    Args:
        model        : the sMLP model being trained
        train_loader : DataLoader yielding (weak_aug, strong_aug) pairs
        optimizer    : Adam optimiser
        scheduler    : linear LR decay scheduler
        criterion    : BCELoss (binary cross-entropy on Hard-LWTA codes)
        epoch        : current epoch index (for logging)

    Returns:
        mean loss over the epoch, current learning rate
    """
    epoch_loss_u   = 0.0
    total_samples  = 0
    model.train()

    # Frozen copy of the model used to generate pseudo-labels.
    # Decoupling pseudo-label generation from the live model avoids
    # unstable feedback loops during training.
    model_backup = CSSLNet().to(device)
    model_backup.load_state_dict(model.state_dict())
    model_backup.eval()

    for _ in tqdm(range(args.n_iters), desc=f"Epoch {epoch+1}", ascii=False, ncols=100):
        for data, _ in train_loader:
            optimizer.zero_grad()

            # data[0]: weakly augmented view  → used for pseudo-label generation
            # data[1]: strongly augmented view → used for loss computation
            ims_u        = data[0].to(device, dtype=torch.float32).reshape(-1, args.c, args.w, args.h)
            ims_u_strong = data[1].to(device, dtype=torch.float32).reshape(-1, args.c, args.w, args.h)

            # Generate binary pseudo-labels from the frozen model (no gradient)
            with torch.no_grad():
                pseudo_label = Hard_LWTA(model_backup(ims_u))   # binary code: (B, L2)

            # Forward pass on strongly augmented view
            scores_strong = model(ims_u_strong)                 # continuous scores: (B, L2)

            # BCE loss: match continuous scores to binary pseudo-labels
            loss = criterion(scores_strong, pseudo_label)

            loss.backward()
            optimizer.step()

            epoch_loss_u  += loss.item() * ims_u_strong.size(0)
            total_samples += ims_u_strong.size(0)

    scheduler.step()
    return epoch_loss_u / total_samples, optimizer.param_groups[0]['lr']


def train(trainset, test_loader, prototypes, model, criterion):
    """
    Full training loop over all epochs.

    At each epoch:
        1. Randomly subsample the training set (controlled by args.split).
        2. Run one epoch of pseudo-label training.
        3. Evaluate on the test set using cosine similarity to prototypes.
        4. Save a checkpoint.

    Returns accumulated metrics and final predictions for ensemble voting.
    """
    train_losses       = []
    acc_simi_max_list  = []
    acc_simi_mean_list = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)

    # Linear LR decay: lr goes from lr_init to lr_init * 0.001 over 100 epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.001,
        total_iters=100, last_epoch=-1, verbose=False
    )

    for epoch in range(args.epochs):
        # Draw a random subset of the training set for this epoch
        len_split    = int(args.split * len(trainset))
        indices      = random.sample(range(len(trainset)), len_split)
        sampler      = torch.utils.data.SubsetRandomSampler(indices)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, sampler=sampler, num_workers=4
        )

        epoch_loss, current_lr = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, epoch
        )
        train_losses.append(epoch_loss)

        sp1, sp2 = calculate_sparsity_rates(model)
        print('| Epoch: {:<4} | LR: {:.6f} | Loss: {:.4f} | s1: {:.3f} | s2: {:.3f} |'.format(
            epoch + 1, current_lr, epoch_loss, sp1, sp2))

        epoch_acc, minidist, preds_max, preds_mean, labels, conf_max, conf_mean = \
            test(model, test_loader, prototypes, epoch)
        acc_simi_max_list.append(epoch_acc)
        acc_simi_mean_list.append(epoch_acc)

        # Save checkpoint at every epoch (overwrite previous)
        # To save only periodically, uncomment the condition below:
        # if (epoch + 1) % 50 == 0:
        folder_path = f'/nasbrain/b22oubah/saved_models/{wandb.run.name}'
        os.makedirs(folder_path, exist_ok=True)
        save_checkpoint(model, optimizer, scheduler, epoch,
                        f'{folder_path}/{seed}.pth')

    return (train_losses, acc_simi_max_list, acc_simi_mean_list,
            preds_max, preds_mean, labels, conf_max, conf_mean, minidist)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def test(model, test_loader, prototypes, epoch, draw_cm=False):
    """
    Evaluates the model on the test set using cosine similarity to class prototypes.

    Inference strategy (unsupervised / prototype-based):
        1. For each class, compute the mean of prototype embeddings (proto_mean).
        2. For each test sample, compute cosine similarity to all class prototypes.
        3. Predict the class with the highest similarity.

    This does not require any labels at test time beyond the prototype set,
    making it compatible with the unsupervised learning setting.

    Args:
        model      : trained CSSLNet
        test_loader: DataLoader for the test set
        prototypes : prototype loader (one or few labelled samples per class)
        epoch      : current epoch (used to collect final-epoch predictions)
        draw_cm    : if True, compute the confusion matrix (not plotted here)

    Returns:
        epoch_acc  : top-1 accuracy
        minidist   : minimum inter-class Hamming distance
        preds_mean : predicted labels at last epoch (for ensemble voting)
        labels_all : ground-truth labels at last epoch
        conf_mean  : normalised similarity scores (soft confidences)
    """
    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    # Compute prototype embeddings and binary codes
    assign_cluster_proba, assign_cluster_binai = assign(model, prototypes[0])
    minidist = hamming_dist(assign_cluster_binai, epoch)

    # Average prototype embeddings per class: list of [L2] tensors
    proto_mean = []
    for key in assign_cluster_proba:
        stacked = torch.stack(assign_cluster_proba[key])   # (n_proto, 1, L2)
        proto_mean.append(stacked.mean(dim=0).squeeze(0))  # → (L2,)

    epoch_acc  = 0.0
    total      = 0
    cm_preds   = torch.tensor([], dtype=torch.long)
    cm_targets = torch.tensor([], dtype=torch.long)
    preds_mean = torch.tensor([], dtype=torch.long)
    labels_all = torch.tensor([], dtype=torch.long)
    conf_mean  = torch.tensor([], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data   = data.to(device, dtype=torch.float32)
            label  = label.to(device, dtype=torch.long)
            output = model(data.reshape(-1, args.c, args.w, args.h))  # (B, L2)

            # Compute cosine similarity between each test sample and each class prototype
            sims = torch.stack([
                torch.stack([cos_sim(output[i], proto_mean[c]) for c in range(10)])
                for i in range(len(output))
            ])  # (B, 10)

            # Nearest-prototype prediction
            # Logit transform (commented out) was tested as an alternative to raw sims
            #logits = torch.logit(sims)
            pred       = sims.argmax(dim=1).to(dtype=torch.long)
            epoch_acc += (label == pred.to(device)).sum().item()
            total     += label.size(0)

            # L1-normalised similarities used as soft confidences for ensemble voting
            conf_norm  = F.normalize(sims, p=1, dim=-1)
            conf_mean  = torch.cat((conf_mean,  conf_norm.cpu()), dim=0)
            cm_targets = torch.cat((cm_targets, label.cpu()),     dim=0)
            cm_preds   = torch.cat((cm_preds,   pred.cpu()),      dim=0)

            # Collect predictions at the final epoch for ensemble voting
            if epoch + 1 == args.epochs:
                preds_mean = torch.cat((preds_mean.to(device), pred.to(device)),  dim=0)
                labels_all = torch.cat((labels_all.to(device), label.to(device)), dim=0)

            if draw_cm:
                cnf_matrix = confusion_matrix(cm_targets, cm_preds)

    epoch_acc = epoch_acc / total
    print('| Similarity-Based Acc: {:.2f}%  | Min Hamming Dist: {} |'.format(
        epoch_acc * 100, int(minidist)))
    wandb.log({f'accuracy_{seed}': epoch_acc})

    return epoch_acc, minidist, preds_mean, preds_mean, labels_all, conf_mean, conf_mean


# ─────────────────────────────────────────────
# Voting / ensemble
# ─────────────────────────────────────────────

def vote(predictions, true_labels):
    """
    Hard majority voting across an ensemble of models.

    Args:
        predictions : tensor of shape (n_models, n_samples) with predicted class indices
        true_labels : tensor of shape (n_samples,) with ground-truth class indices

    Returns:
        Scalar accuracy of the ensemble majority vote.
    """
    prediction_vote = torch.mode(predictions, dim=0)[0]   # majority class per sample
    return (prediction_vote == true_labels).sum() / true_labels.shape[0]


def vote_proba(confidences, true_labels):
    """
    Soft probability voting across an ensemble of models.

    Averages the L1-normalised similarity scores across models and
    predicts the class with the highest cumulative score.

    Args:
        confidences : tensor of shape (n_models * n_samples, 10)
        true_labels : tensor of shape (n_samples,) with ground-truth class indices

    Returns:
        Scalar accuracy of the soft ensemble vote.
    """
    confidences = confidences.reshape(-1, true_labels.shape[0], 10)  # (n_models, n_samples, 10)
    prob_sum    = confidences.sum(0)                                  # (n_samples, 10)
    pred_label  = prob_sum.argmax(1).to(dtype=torch.long)
    return (pred_label == true_labels).sum() / true_labels.shape[0]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    """
    Entry point. Runs training and evaluation for multiple values of gamma.

    Currently gamma is passed to args but not used in the loss (kept for
    ablation experiments). Multiple seeds can be enabled by extending the
    seeds list.

    Ensemble voting is performed after each new model is trained, accumulating
    both hard (majority) and soft (probability) votes.
    """
    global seed

    os.makedirs("./results/", exist_ok=True)

    seeds  = [3124]  # extend to [3124, 2023, 31478, ...] for multi-seed runs
    gammas = [0.1, 0.3, 0.5, 2]  # ablation values for the gamma hyperparameter

    # Accumulators for ensemble voting across trained models
    predictions_mean  = torch.tensor([], device=device, dtype=torch.long)
    confidences_mean  = torch.tensor([], device=device, dtype=torch.float32)

    seed_acc_list  = []
    seed_dist_list = []

    for seed_idx, seed in enumerate(seeds):

        # --- Reproducibility ---
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        # --- Data ---
        trainset, proto_loader, test_loader = get_data_loader(
            batch_size=args.batch_size, seed=seed
        )

        # --- Model & criterion ---
        model     = CSSLNet().apply(init_weights).to(device)
        criterion = nn.BCELoss()  # binary cross-entropy between continuous scores and binary pseudo-labels

        # --- Train ---
        (train_losses, acc_max_list, acc_mean_list,
         preds_max, preds_mean, labels_vote,
         conf_max, conf_mean, minidist) = train(
            trainset, test_loader, proto_loader, model, criterion
        )

        seed_acc_list.append(acc_mean_list[-1])
        seed_dist_list.append(minidist)

        # --- Accumulate ensemble predictions ---
        preds_mean        = preds_mean.reshape(1, -1).to(device)
        predictions_mean  = torch.cat((predictions_mean, preds_mean),            dim=0)
        confidences_mean  = torch.cat((confidences_mean, conf_mean.to(device)),  dim=0)

        # --- Ensemble voting ---
        vote_acc  = round(vote(predictions_mean, labels_vote).item() * 100, 2)
        proba_acc = round(vote_proba(confidences_mean, labels_vote).item() * 100, 2)

        print(f"|--- Ensemble ({seed_idx+1} networks) ---|")
        print(f"| Majority vote acc : {vote_acc:.2f}%  |  Proba vote acc : {proba_acc:.2f}% |")

    wandb.finish()


if __name__ == '__main__':
    main()
