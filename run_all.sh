#!/usr/bin/env bash
# Full experiment: baselines + 4-step ablation ladder + eval + table + curves.
# Usage: bash run_all.sh            (1 seed)    |    SEEDS="0 1 2" bash run_all.sh   (mean±std)
set -e
SEEDS=${SEEDS:-"42"}
FIRST=$(echo $SEEDS | awk '{print $1}')

python baseline.py --seed 42
for s in $SEEDS; do
  python train.py --name plain   --seed $s --label-smoothing 0 --ema 0 --mix none     # CNN + aug + dropout + wd
  python train.py --name tricks  --seed $s --mix none                                 # + label smoothing + EMA
  python train.py --name mixup   --seed $s --mix mixup                                # + standard mixup
  python train.py --name confmix --seed $s --mix confmix                              # + confusion-guided mixup (ours)
  for n in plain tricks mixup confmix; do python eval.py --run runs/${n}_s${s}; done
done
python summarize.py
python plot_curves.py --runs runs/plain_s${FIRST} runs/tricks_s${FIRST} runs/mixup_s${FIRST} runs/confmix_s${FIRST} --out figures/curves.png
