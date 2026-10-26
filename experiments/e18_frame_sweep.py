"""
E18: Is the joke-frame benefit an artifact of one prompt wording?
==================================================================
Sweeps five semantically equivalent joke/factual frame pairs over the novel
jokes (memorisation-proof, see E16) vs prose controls, and checks whether the
joke-frame benefit separates under every phrasing.

Model: gemma-2-2b-it. Outputs: results/frame_sweep_results.csv, fig_frame_sweep.png
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from humour.corpus import CONTROLS, NOVEL_JOKES
from humour.plotting import results_path, save
from humour.stats import compare
from humour.surprisal import load, punchline_surprisal

MODEL_NAME = "google/gemma-2-2b-it"
FRAME_PAIRS = [
    ("wellknown", "Here is a well-known joke: ", "Here is a routine factual note: "),
    ("following", "The following text is a joke.\n", "The following text is a factual statement.\n"),
    ("terse", "Joke:\n", "Note:\n"),
    ("funny", "What follows is meant to be funny. ", "What follows is meant to inform. "),
    ("reader", "You are about to read a joke. ", "You are about to read a factual memo. "),
]
CORPUS = [(n, "joke", s, p) for n, _, s, p in NOVEL_JOKES] + CONTROLS

model, tokenizer = load(MODEL_NAME)

rows, summary = [], []
for frame_name, frame_joke, frame_memo in FRAME_PAIRS:
    benefits = {"joke": [], "control": []}
    for name, cat, setup, punch in CORPUS:
        b = (punchline_surprisal(model, tokenizer, frame_memo, setup, punch)
             - punchline_surprisal(model, tokenizer, frame_joke, setup, punch))
        benefits[cat].append(b)
        rows.append([frame_name, name, cat, f"{b:.3f}"])
    c = compare(benefits["joke"], benefits["control"])
    summary.append((frame_name, np.median(benefits["joke"]), np.median(benefits["control"]), c["p"]))
    print(f"{frame_name:<11} joke median={np.median(benefits['joke']):+5.2f}  "
          f"control median={np.median(benefits['control']):+5.2f}  U={c['u']:.1f}  "
          f"p={c['p']:.4f}  delta={c['delta']:+.2f}", flush=True)

with open(results_path("frame_sweep_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_pair", "name", "category", "joke_frame_benefit"])
    w.writerows(rows)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(summary))
joke_medians = [s[1] for s in summary]
control_medians = [s[2] for s in summary]
ax.bar(x - 0.18, joke_medians, width=0.36, color="#d1495b", label="Novel jokes (median)")
ax.bar(x + 0.18, control_medians, width=0.36, color="#8d99ae", label="Prose controls (median)")
for i, (_, jm, cm, p) in enumerate(summary):
    ax.text(i, max(jm, cm) + 0.04, f"p={p:.3f}", ha="center", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in summary])
ax.set_ylabel("Joke-frame benefit (bits/token)")
ax.set_title("gemma-2-2b-it: joke-frame benefit across five frame phrasings (novel jokes vs prose)", fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_frame_sweep.png")
