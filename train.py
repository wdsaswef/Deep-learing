"""Train the CNN on Fashion-MNIST. NEVER touches the test set (that is eval.py's job).

Distinctive part: --mix confmix  ("confusion-guided mixup")
  1. Train a quick pilot CNN for a few epochs (no mixing).
  2. Build a confusion matrix on the VALIDATION set -> which classes get mixed up?
  3. In the real run, each image is blended with an image from a class it is
     OFTEN CONFUSED WITH (e.g. Shirt <-> T-shirt <-> Pullover <-> Coat),
     with a soft label. The model is forced to learn the fine boundary exactly
     where it is weakest.
Other recipes: --mix mixup (standard), --mix none.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from models import SmallCNN
from utils import (CLASSES, augment_batch, confusion_matrix, get_device,
                   load_data, metrics_from_logits, predict_logits, set_seed)


# ------------------------------------------------------------------ confusion-guided partner sampling
class ClassSampler:
    """Given a class y, sample a partner class c ~ P[y] (more likely = more confused with y),
    then a random training image of class c."""

    def __init__(self, x, y, confusion, eps=0.02, n=10):
        self.x = x
        self.order = torch.argsort(y)
        self.counts = torch.bincount(y, minlength=n)
        self.starts = torch.cumsum(self.counts, 0) - self.counts
        off = confusion.float().to(x.device)
        off.fill_diagonal_(0)                                   # only mistakes
        off = off / off.sum(1, keepdim=True).clamp(min=1)       # row-normalise
        P = off + eps                                           # eps = a little exploration
        P.fill_diagonal_(0)
        self.P = P / P.sum(1, keepdim=True)

    def partner(self, y):
        cls = torch.multinomial(self.P[y], 1).squeeze(1)
        r = (torch.rand(len(y), device=y.device) * self.counts[cls]).long()
        r = torch.minimum(r, self.counts[cls] - 1)
        return self.x[self.order[self.starts[cls] + r]], cls


def make_mix(xb, yb, cfg, sampler):
    """Returns mixed images, partner labels, lambda (per-sample, >=0.5 so the original dominates)."""
    B = xb.size(0)
    if cfg.mix == "mixup":
        j = torch.randperm(B, device=xb.device)
        x2, y2 = xb[j], yb[j]
    else:  # confmix
        x2, y2 = sampler.partner(yb)
        x2 = augment_batch(x2, cfg.pad, cfg.flip)
    a = torch.tensor(cfg.alpha)
    lam = torch.distributions.Beta(a, a).sample((B,)).to(xb.device)
    lam = torch.maximum(lam, 1 - lam)
    use = torch.rand(B, device=xb.device) < cfg.mix_prob
    lam = torch.where(use, lam, torch.ones_like(lam))
    xm = lam.view(-1, 1, 1, 1) * xb + (1 - lam.view(-1, 1, 1, 1)) * x2
    return xm, y2, lam


@torch.no_grad()
def update_ema(ema, model, decay, step):
    d = min(decay, (1 + step) / (10 + step))   # warm-up so early EMA isn't dominated by random init
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(d).add_(p.detach(), alpha=1 - d)
    for be, b in zip(ema.buffers(), model.buffers()):   # BatchNorm running stats
        be.copy_(b)


# ------------------------------------------------------------------ training loop
def fit(cfg, data, device, out_dir=None, confusion=None, verbose=True):
    set_seed(cfg.seed)
    xtr, ytr, xva, yva = data["x_train"], data["y_train"], data["x_val"], data["y_val"]

    model = SmallCNN(cfg.width, cfg.dropout).to(device)
    ema = None
    if cfg.ema > 0:
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    spe = len(xtr) // cfg.bs
    sched = OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * spe, pct_start=0.25)
    sampler = ClassSampler(xtr, ytr, confusion, cfg.conf_eps) if cfg.mix == "confmix" else None

    hist = {k: [] for k in ["epoch", "lr", "train_loss", "val_loss", "val_acc", "train_acc_clean"]}
    best_acc, best_state, bad, step = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        run = torch.zeros((), device=device)
        for i in range(spe):
            idx = perm[i * cfg.bs:(i + 1) * cfg.bs]
            xb, yb = xtr[idx], ytr[idx]
            xb = augment_batch(xb, cfg.pad, cfg.flip)
            if cfg.mix != "none":
                xb, y2, lam = make_mix(xb, yb, cfg, sampler)
                logits = model(xb)
                l1 = F.cross_entropy(logits, yb, label_smoothing=cfg.label_smoothing, reduction="none")
                l2 = F.cross_entropy(logits, y2, label_smoothing=cfg.label_smoothing, reduction="none")
                loss = (lam * l1 + (1 - lam) * l2).mean()
            else:
                loss = F.cross_entropy(model(xb), yb, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if ema is not None:
                update_ema(ema, model, cfg.ema, step)
            run += loss.detach()

        em = ema if ema is not None else model
        v = metrics_from_logits(predict_logits(em, xva), yva)          # plain CE, no smoothing
        tr = metrics_from_logits(predict_logits(em, xtr[:5000]), ytr[:5000])  # clean train acc (no aug)
        for k, val in zip(hist, [epoch, sched.get_last_lr()[0], (run / spe).item(),
                                 v["loss"], v["acc"], tr["acc"]]):
            hist[k].append(val)
        if verbose:
            print(f"[{cfg.name}] ep {epoch:02d}/{cfg.epochs} lr {hist['lr'][-1]:.4f} "
                  f"train_loss {hist['train_loss'][-1]:.4f} val_loss {v['loss']:.4f} "
                  f"val_acc {v['acc']:.4f} train_acc {tr['acc']:.4f} ({time.time()-t0:.0f}s)")

        if v["acc"] > best_acc:
            best_acc, bad = v["acc"], 0
            best_state = copy.deepcopy(em.state_dict())
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    em = ema if ema is not None else model
    em.load_state_dict(best_state)
    if out_dir is not None:
        torch.save(best_state, Path(out_dir) / "best.pt")
    return em, hist, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42, help="fixed -> same train/val split for every run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3, help="OneCycle max LR")
    ap.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pad", type=int, default=2, help="random-translation pixels")
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay; 0 = off")
    ap.add_argument("--mix", choices=["none", "mixup", "confmix"], default="none")
    ap.add_argument("--mix-prob", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.4, help="Beta(alpha, alpha) for lambda")
    ap.add_argument("--conf-eps", type=float, default=0.02)
    ap.add_argument("--pilot-epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10, help="early stopping on val acc; 0 = off")
    ap.add_argument("--data-root", default="data")
    cfg = ap.parse_args()

    device = get_device()
    data = load_data(cfg.data_root, cfg.split_seed, device=device)
    out = Path("runs") / f"{cfg.name}_s{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | train {len(data['x_train'])} val {len(data['x_val'])} | out={out}")

    confusion = None
    if cfg.mix == "confmix":
        print(f"--- pilot run ({cfg.pilot_epochs} epochs) to find confused class pairs (val set only) ---")
        pc = argparse.Namespace(**{**vars(cfg), "epochs": cfg.pilot_epochs, "mix": "none",
                                   "ema": 0.0, "name": "pilot", "patience": 0})
        pm, _, pacc = fit(pc, data, device, verbose=False)
        confusion = confusion_matrix(predict_logits(pm, data["x_val"]).argmax(1), data["y_val"])
        off = confusion.clone(); off.fill_diagonal_(0)
        print(f"pilot val acc {pacc:.4f}. Most confused pairs (true -> predicted : count):")
        for k in torch.topk(off.flatten(), 5).indices.tolist():
            print(f"   {CLASSES[k // 10]} -> {CLASSES[k % 10]} : {off[k // 10, k % 10].item()}")
        json.dump(confusion.tolist(), open(out / "pilot_confusion.json", "w"))

    model, hist, best = fit(cfg, data, device, out_dir=out, confusion=confusion)
    json.dump(vars(cfg), open(out / "config.json", "w"), indent=2)
    json.dump(hist, open(out / "history.json", "w"))
    print(f"done. best val acc = {best:.4f}  -> now run: python eval.py --run {out}")


if __name__ == "__main__":
    main()
