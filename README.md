# Confusion-Guided Mixup for Fashion-MNIST (Deep Learning Capstone)

A small CNN trained from scratch on **Fashion-MNIST**, compared against a majority-class and a
logistic-regression baseline, with a 4-step ablation that ends in a lightly novel training recipe:
**confusion-guided mixup**.

> **TODO before submitting:** replace every `TODO` below with your real numbers/links (run the code first!).

## 1. Task, data, metric
| | |
|---|---|
| Task | 10-class clothing image classification (28×28 grayscale) |
| Dataset | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — Xiao, Rasul & Vollgraf, 2017 (auto-downloaded by `torchvision`, not committed) |
| Splits | official 60k train → **55,000 train / 5,000 val** (fixed split seed 42, identical for every run); official **10,000 test** used only in `baseline.py` / `eval.py` |
| Metrics | accuracy (main), macro-F1 (class-balanced view), test cross-entropy |
| Baselines | (a) majority class, (b) logistic regression (single linear layer + softmax) |

## 2. Model & training recipe
**Model** (`models.py`, ~140k params): `[Conv3×3-BN-ReLU]×2 → MaxPool → [Conv-BN-ReLU]×2 → MaxPool → Conv-BN-ReLU → GlobalAvgPool → Dropout(0.3) → Linear(10)`.

**Training essentials:** cross-entropy loss, AdamW (weight decay 0.05), One-Cycle LR schedule (max 3e-3), batch 128, 30 epochs, early stopping/checkpoint on **validation** accuracy, seed 42.

**Regularization:** dropout, weight decay, data augmentation (random ±2px translation + horizontal flip), label smoothing (0.1), EMA of weights, and mixup variants.

### The distinctive part: confusion-guided mixup (`--mix confmix`)
Standard mixup blends each image with a *random* one. Most random pairs (e.g. *Sandal + Trouser*) are already trivially separable, so the regularizer is mostly wasted. Instead:
1. Train a short **pilot** CNN (4 epochs) and build a **confusion matrix on the validation set**.
2. For each training image of class *y*, pick a partner class with probability ∝ how often *y* is confused with it (plus small ε exploration), then blend with a random image of that class (λ ~ Beta(0.4,0.4), kept ≥ 0.5) and use the soft label.
3. The model therefore practises on the *hard boundaries* (Shirt ↔ T-shirt ↔ Pullover ↔ Coat).

### Ablation ladder (`run_all.sh`)
| name | what is added |
|---|---|
| `plain` | CNN + augmentation + dropout + weight decay + One-Cycle |
| `tricks` | `plain` + label smoothing + EMA weights |
| `mixup` | `tricks` + standard mixup |
| `confmix` | `tricks` + confusion-guided mixup (ours) |

## 3. Results (test set, seed 42)
`TODO: paste results_table.md here after running run_all.sh`

| Model | Test acc (%) | Macro-F1 (%) |
|---|---|---|
| majority | TODO | TODO |
| logistic regression | TODO | TODO |
| plain / tricks / mixup / confmix | TODO | TODO |

Figures: `figures/curves.png` (learning curves), `runs/<run>/confusion_matrix.png`, `runs/<run>/failure_cases.png`.

**Honest analysis (TODO):** does confmix actually beat mixup/tricks beyond seed noise? (run `SEEDS="0 1 2 3 4" bash run_all.sh` for mean ± std.) Which classes are still confused and why? See `failure_cases.png`.

## 4. Reproduce
```bash
pip install -r requirements.txt
bash run_all.sh                      # baselines + 4 recipes + eval + table + curves (≈ 10–20 min on a Colab T4)
# multiple seeds:  SEEDS="0 1 2" bash run_all.sh

# or step by step
python baseline.py
python train.py --name confmix --mix confmix       # writes runs/confmix_s42/{best.pt,config.json,history.json}
python eval.py  --run runs/confmix_s42             # the ONLY place (besides baseline.py) that reads the test set
python summarize.py
python plot_curves.py --runs runs/plain_s42 runs/confmix_s42
```
Quick smoke test without downloading data (random tensors, checks the code runs only):
`FASHION_FAKE=1 python train.py --name smoke --epochs 2 --pilot-epochs 1 --mix confmix && FASHION_FAKE=1 python eval.py --run runs/smoke_s42`

## 5. Methodology / no leakage
- Test set is never loaded by `train.py`; hyper-parameters and checkpoints are chosen on the validation set only.
- The confusion matrix used by confmix comes from the **validation** set and a pilot model.
- Normalization uses fixed constants (mean 0.2860, std 0.3530), not test statistics.
- Fixed seeds (`set_seed`, deterministic cuDNN). Tiny GPU nondeterminism may still shift results by a few hundredths of a percent.

## 6. Repo layout
```
utils.py      seeding, data split, GPU augmentation, metrics
models.py     SmallCNN + LinearBaseline
baseline.py   majority + logistic regression
train.py      training loop (One-Cycle, EMA, mixup/confmix)
eval.py       test metrics, flip-TTA, confusion matrix, failure cases
summarize.py  results table      plot_curves.py  learning curves
run_all.sh    full experiment
```

## 7. Limitations / next steps (TODO: adapt to your findings)
- Fashion-MNIST is small and nearly saturated; gains of ~0.1–0.3 pts are comparable to seed noise → needs multiple seeds.
- Confusion matrix is computed once from a weak pilot model; it could be refreshed every few epochs.
- Try harder data (CIFAR-10) or a ResNet-style model.

## 8. Demo
`TODO: Colab / video link`
