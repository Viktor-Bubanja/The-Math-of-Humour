"""
E17: Does "deliberate nonsense" work as its own interpretation?
================================================================
Absurdist anti-humour arguably still has a second interpretation, one level up:
"this is a deliberate performance of incoherence". If so, declaring that frame
should compress absurdist punchlines better than a plain "joke" frame, while
ordinary jokes should prefer the joke frame.

Corpus: 10 absurdist texts (5 classic, 5 new), the 12 novel jokes from E16,
and the 10 prose controls. Three uniform frames: absurd / joke / memo.

Model: gemma-2-2b-it (the meta-frame wording needs some semantic capability).
Outputs: results/absurdism_frame_results.csv, fig_absurdism_frame.png
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from humour.corpus import ABSURDIST, CONTROLS, NOVEL_JOKES
from humour.plotting import results_path, save, strip_plot
from humour.stats import compare
from humour.surprisal import load, punchline_surprisal

MODEL_NAME = "google/gemma-2-2b-it"
FRAMES = {
    "absurd": "The following text is absurdist anti-humour, a deliberate performance of nonsense.\n",
    "joke": "The following text is a joke.\n",
    "memo": "The following text is a factual statement.\n",
}
CATS = ("absurdist", "joke", "control")
CORPUS = ([(n, "absurdist", s, p) for n, s, p in ABSURDIST]
          + [(n, "joke", s, p) for n, _, s, p in NOVEL_JOKES]
          + CONTROLS)

model, tokenizer = load(MODEL_NAME)

results = []
for name, cat, setup, punch in CORPUS:
    surps = {f: punchline_surprisal(model, tokenizer, prefix, setup, punch) for f, prefix in FRAMES.items()}
    best = min(surps, key=lambda f: surps[f])
    results.append(dict(name=name, category=cat, best=best, **surps))
    print(f"  {name:<16} {cat:<10} " + "  ".join(f"{f}={s:5.2f}" for f, s in surps.items())
          + f"  best={best}", flush=True)


def by_cat(cat):
    return [r for r in results if r["category"] == cat]


print("\nBest-compressing frame (count of texts):")
for cat in CATS:
    counts = {f: sum(r["best"] == f for r in by_cat(cat)) for f in FRAMES}
    print(f"  {cat:<10} " + "  ".join(f"{f}: {c}" for f, c in counts.items()))

print("\nAbsurd-frame advantage over joke frame (joke surprisal - absurd surprisal):")
advantage = {cat: [r["joke"] - r["absurd"] for r in by_cat(cat)] for cat in CATS}
for cat in CATS:
    print(f"  {cat:<10} median={np.median(advantage[cat]):+6.2f}")
c = compare(advantage["absurdist"], advantage["joke"])
print(f"  absurdist vs joke texts: U={c['u']:.1f}  p={c['p']:.4f}  delta={c['delta']:+.2f}")

with open(results_path("absurdism_frame_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "category", "surprisal_absurd_frame", "surprisal_joke_frame",
                "surprisal_memo_frame", "best_frame"])
    for r in results:
        w.writerow([r["name"], r["category"], f"{r['absurd']:.3f}", f"{r['joke']:.3f}",
                    f"{r['memo']:.3f}", r["best"]])

fig, ax = plt.subplots(figsize=(7.5, 4.8))
strip_plot(ax, advantage, labels=["Absurdist", "Novel jokes", "Prose controls"])
ax.set_ylabel("Absurd-frame advantage (bits/token)\njoke-frame surprisal − absurd-frame surprisal")
ax.set_title('gemma-2-2b-it: does "deliberate nonsense" out-compress "joke" on absurdist texts?', fontsize=10)
save(fig, "fig_absurdism_frame.png")
