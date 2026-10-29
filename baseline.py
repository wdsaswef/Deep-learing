"""Baselines: (1) majority-class predictor, (2) logistic regression (linear softmax).
Test set is touched only once at the very end."""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models import LinearBaseline
from utils import (confusion_matrix, get_device, load_data, macro_f1,
                   metrics_from_logits, predict_logits, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = get_device()
    d = load_data(args.data_root, args.split_seed, device=dev)
    xtr, ytr, xva, yva, xte, yte = (d[k] for k in
        ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"])

    # ---- 1) majority class ----
    maj = torch.bincount(ytr).argmax()
    pred = torch.full_like(yte, maj.item())
    conf = confusion_matrix(pred, yte)
    mf1, _ = macro_f1(conf)
    res = dict(name="baseline_majority", seed=args.seed,
               test_acc=(pred == yte).float().mean().item(), test_macro_f1=mf1)
    out = Path("runs/baseline_majority_s%d" % args.seed); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "results.json", "w"), indent=2)
    print("majority:", res)

    # ---- 2) logistic regression ----
    model = LinearBaseline().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_acc, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), args.bs):
            idx = perm[i:i + args.bs]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m = metrics_from_logits(predict_logits(model, xva), yva)
        if m["acc"] > best_acc:
            best_acc, best_state = m["acc"], copy.deepcopy(model.state_dict())
        print(f"[logreg] epoch {ep:02d} val_acc {m['acc']:.4f}")
    model.load_state_dict(best_state)
    t = metrics_from_logits(predict_logits(model, xte), yte)
    res = dict(name="baseline_logreg", seed=args.seed, val_acc=best_acc,
               test_acc=t["acc"], test_macro_f1=t["macro_f1"], test_loss=t["loss"])
    out = Path("runs/baseline_logreg_s%d" % args.seed); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "results.json", "w"), indent=2)
    print("logreg:", res)


if __name__ == "__main__":
    main()
