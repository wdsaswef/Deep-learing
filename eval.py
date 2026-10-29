"""Final evaluation on the held-out TEST set + confusion matrix + failure cases."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from models import SmallCNN
from utils import CLASSES, MEAN, STD, get_device, load_data, metrics_from_logits, predict_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. runs/confmix_s42")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = json.load(open(run / "config.json"))

    dev = get_device()
    data = load_data(args.data_root, cfg["split_seed"], device=dev)
    xte, yte = data["x_test"], data["y_test"]
    model = SmallCNN(cfg["width"], cfg["dropout"]).to(dev)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))

    logits = predict_logits(model, xte)
    m = metrics_from_logits(logits, yte)
    m_tta = metrics_from_logits(predict_logits(model, xte, tta=True), yte)

    conf = m["confusion"].cpu()
    off = conf.clone(); off.fill_diagonal_(0)
    top = [dict(true=CLASSES[k // 10], pred=CLASSES[k % 10], count=int(off[k // 10, k % 10]))
           for k in torch.topk(off.flatten(), 6).indices.tolist()]
    res = dict(name=cfg["name"], seed=cfg["seed"],
               test_acc=m["acc"], test_macro_f1=m["macro_f1"], test_loss=m["loss"],
               test_acc_tta=m_tta["acc"], test_macro_f1_tta=m_tta["macro_f1"],
               per_class_f1=dict(zip(CLASSES, m["per_class_f1"])), top_confusions=top)
    json.dump(res, open(run / "results.json", "w"), indent=2)
    print(f"TEST  acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | loss {m['loss']:.4f}"
          f"  ||  with flip-TTA: acc {m_tta['acc']:.4f} | macro-F1 {m_tta['macro_f1']:.4f}")
    print("Top confusions:", top)

    # ---- confusion matrix (row-normalised) ----
    norm = conf.float() / conf.sum(1, keepdim=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=60, ha="right"); ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Test confusion matrix - {run.name}")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(run / "confusion_matrix.png", dpi=150); plt.close(fig)

    # ---- failure cases: most confidently WRONG test images ----
    prob = F.softmax(logits, 1)
    conf_max, pred = prob.max(1)
    wrong = (pred != yte).nonzero().squeeze(1)
    if len(wrong):
        pick = wrong[torch.topk(conf_max[wrong], min(12, len(wrong))).indices]
        fig, axes = plt.subplots(3, 4, figsize=(9, 8))
        for a in axes.flat:
            a.axis("off")
        for a, i in zip(axes.flat, pick.tolist()):
            a.imshow((xte[i, 0].cpu() * STD + MEAN), cmap="gray")
            a.set_title(f"true: {CLASSES[yte[i]]}\npred: {CLASSES[pred[i]]} ({conf_max[i]:.2f})", fontsize=8)
        fig.suptitle("Most confident mistakes (test set)"); fig.tight_layout()
        fig.savefig(run / "failure_cases.png", dpi=150); plt.close(fig)
    print(f"saved results.json, confusion_matrix.png, failure_cases.png in {run}")


if __name__ == "__main__":
    main()
