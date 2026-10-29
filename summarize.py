"""Collect runs/*/results.json -> markdown table (mean +- std over seeds)."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

rows = defaultdict(list)
for p in sorted(Path("runs").glob("*/results.json")):
    r = json.load(open(p))
    rows[r["name"]].append(r)

def fmt(vals):
    if not vals:
        return "-"
    m = 100 * st.mean(vals)
    return f"{m:.2f}" if len(vals) == 1 else f"{m:.2f} ± {100 * st.stdev(vals):.2f}"

order = ["baseline_majority", "baseline_logreg", "plain", "tricks", "mixup", "confmix"]
names = [n for n in order if n in rows] + [n for n in rows if n not in order]
lines = ["| Model | #seeds | Test acc (%) | Test macro-F1 (%) | Test acc + flip-TTA (%) |",
         "|---|---|---|---|---|"]
for n in names:
    rs = rows[n]
    lines.append(f"| {n} | {len(rs)} | {fmt([r['test_acc'] for r in rs])} | "
                 f"{fmt([r['test_macro_f1'] for r in rs])} | "
                 f"{fmt([r['test_acc_tta'] for r in rs if 'test_acc_tta' in r])} |")
table = "\n".join(lines)
print(table)
Path("results_table.md").write_text(table + "\n")
