"""Plot learning curves for one or more runs (overlay for ablations)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", default="figures/curves.png")
args = ap.parse_args()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, r in enumerate(map(Path, args.runs)):
    h = json.load(open(r / "history.json"))
    c, e = colors[i % len(colors)], h["epoch"]
    ax[0].plot(e, h["train_loss"], "--", color=c, label=f"{r.name} train")
    ax[0].plot(e, h["val_loss"], "-", color=c, label=f"{r.name} val")
    ax[1].plot(e, [100 * a for a in h["val_acc"]], color=c, label=r.name)
    ax[2].plot(e, [100 * (t - v) for t, v in zip(h["train_acc_clean"], h["val_acc"])], color=c, label=r.name)
ax[0].set_title("Loss (train = with aug/mix/smoothing; val = plain CE)")
ax[1].set_title("Validation accuracy (%)")
ax[2].set_title("Generalisation gap: clean train acc - val acc (pts)")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=7)
ax[1].set_ylim(bottom=max(80, ax[1].get_ylim()[0]))
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150)
print("saved", args.out)
