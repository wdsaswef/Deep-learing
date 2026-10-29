"""Shared helpers: seeding, data loading/splitting, metrics, GPU augmentation."""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530  # Fashion-MNIST train-set pixel statistics (fixed constants)


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(root="data", split_seed=42, val_size=5000, device="cpu"):
    """Returns whole dataset as normalized tensors on `device`.

    Split: official 60k train -> (60k - val_size) train + val_size val using a
    FIXED split_seed (same split for every run). Official 10k test is untouched.
    Set env FASHION_FAKE=1 for a tiny random dataset (smoke tests only).
    """
    if os.environ.get("FASHION_FAKE") == "1":
        g = torch.Generator().manual_seed(0)
        x = torch.rand(6000, 1, 28, 28, generator=g)
        y = torch.randint(0, 10, (6000,), generator=g)
        xt = torch.rand(1000, 1, 28, 28, generator=g)
        yt = torch.randint(0, 10, (1000,), generator=g)
        val_size = min(val_size, 1000)
    else:
        from torchvision import datasets
        tr = datasets.FashionMNIST(root, train=True, download=True)
        te = datasets.FashionMNIST(root, train=False, download=True)
        x = tr.data.float().div(255).unsqueeze(1)
        y = tr.targets.long()
        xt = te.data.float().div(255).unsqueeze(1)
        yt = te.targets.long()

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(x), generator=g)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    norm = lambda t: (t - MEAN) / STD
    d = dict(x_train=norm(x[tr_idx]), y_train=y[tr_idx],
             x_val=norm(x[val_idx]), y_val=y[val_idx],
             x_test=norm(xt), y_test=yt)
    return {k: v.to(device) for k, v in d.items()}


def augment_batch(x, pad=2, flip=True):
    """Per-sample random translation (replicate-pad + random crop) and horizontal flip.
    Fully vectorized, runs on the same device as x."""
    B, _, H, W = x.shape
    if pad > 0:
        xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        dy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        dx = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        iy = dy[:, None] + torch.arange(H, device=x.device)[None, :]   # (B,H)
        ix = dx[:, None] + torch.arange(W, device=x.device)[None, :]   # (B,W)
        b = torch.arange(B, device=x.device)[:, None, None]
        x = xp[b, 0, iy[:, :, None], ix[:, None, :]].unsqueeze(1)
    if flip:
        m = torch.rand(B, device=x.device) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(3), x)
    return x


@torch.no_grad()
def predict_logits(model, x, bs=1000, tta=False):
    """Eval-mode logits. tta=True averages logits of the image and its horizontal flip."""
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        lo = model(xb)
        if tta:
            lo = (lo + model(xb.flip(3))) / 2
        out.append(lo)
    return torch.cat(out)


def confusion_matrix(pred, target, n=10):
    """Rows = true class, cols = predicted class."""
    return torch.bincount(target * n + pred, minlength=n * n).view(n, n)


def macro_f1(conf):
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    return f1.mean().item(), f1


def metrics_from_logits(logits, y):
    pred = logits.argmax(1)
    conf = confusion_matrix(pred, y)
    mf1, per_class = macro_f1(conf)
    return dict(loss=F.cross_entropy(logits, y).item(),
                acc=(pred == y).float().mean().item(),
                macro_f1=mf1, per_class_f1=per_class.tolist(), confusion=conf)
