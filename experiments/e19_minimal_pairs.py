"""
E19: Does the joke frame reward resolution, or any surprising continuation?
============================================================================
E14-E18 show that a "this is a joke" frame makes joke punchlines cheaper than
prose. But the controls are bland prose, so the frame might help *any*
surprising or non-literal continuation, not specifically one that resolves a
joke. This keeps each joke's setup and swaps the punchline for matched foils
(humour/corpus.py FOILS, fixed and reviewed before this script was first run):

  - real:         the actual punchline
  - incongruous:  same length and syntactic frame, at least as surprising with
                  no frame, but with no second reading that makes the setup fit
  - literal:      the ordinary continuation (reference point, not matched)

benefit = S(punch | memo frame + setup) - S(punch | joke frame + setup), as in
E14/E18, for all 39 jokes (27 famous, 12 novel) x 3 punchline types.

Analysis, fixed before running:
  - Primary: gemma-2-2b-it, "reader" frame pair, all 39 jokes. Paired difference
    benefit(real) - benefit(incongruous) per joke; two-sided Wilcoxon signed-rank,
    median difference, matched-pairs rank-biserial r, 95% bootstrap CI over jokes.
  - Famous and novel jokes also reported separately. Famous real punchlines are
    probably memorised, so the novel subset (n = 12) is the clean test.
  - Secondary: real vs literal (same test) and the median benefit of each type.
  - Robustness: all five E18 phrasings, and distilgpt2.
  - No exclusions: every joke and foil is scored and reported.

Prediction: the incongruity-resolution account predicts a positive real -
incongruous difference. A null means the frame rewards incongruity in general,
not resolution.

Models: distilgpt2, gemma-2-2b-it (CPU, like E11-E18).
Outputs: results/minimal_pairs_results.csv, fig_minimal_pairs.png
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from humour.corpus import FAMOUS_JOKES, FOILS, NOVEL_JOKES
from humour.plotting import COLORS, results_path, save
from humour.stats import paired_compare
from humour.surprisal import load, punchline_surprisal

MODELS = ["distilgpt2", "google/gemma-2-2b-it"]
FRAME_PAIRS = [
    ("reader", "You are about to read a joke. ", "You are about to read a factual memo. "),
    ("wellknown", "Here is a well-known joke: ", "Here is a routine factual note: "),
    ("following", "The following text is a joke.\n", "The following text is a factual statement.\n"),
    ("terse", "Joke:\n", "Note:\n"),
    ("funny", "What follows is meant to be funny. ", "What follows is meant to inform. "),
]
TYPES = ("real", "incongruous", "literal")
JOKES = ([(n, "famous", s, p) for n, _, s, p in FAMOUS_JOKES]
         + [(n, "novel", s, p) for n, _, s, p in NOVEL_JOKES])
SUBSETS = {"all": ("famous", "novel"), "famous": ("famous",), "novel": ("novel",)}

rows = []
for model_name in MODELS:
    print(f"\n=== {model_name} ===", flush=True)
    model, tokenizer = load(model_name)
    for name, source, setup, real in JOKES:
        punches = dict(zip(TYPES, (real, *FOILS[name])))
        for ptype, punch in punches.items():
            s_base = punchline_surprisal(model, tokenizer, "", setup, punch)
            n_tokens = len(tokenizer(punch, add_special_tokens=False)["input_ids"])
            for frame_name, frame_joke, frame_memo in FRAME_PAIRS:
                s_joke = punchline_surprisal(model, tokenizer, frame_joke, setup, punch)
                s_memo = punchline_surprisal(model, tokenizer, frame_memo, setup, punch)
                rows.append(dict(model=model_name, frame=frame_name, name=name, source=source,
                                 type=ptype, punchline=punch, n_tokens=n_tokens, s_base=s_base,
                                 s_joke=s_joke, s_memo=s_memo, benefit=s_memo - s_joke))
        print(f"  {name}", flush=True)


def benefits(model_name, frame, ptype, sources):
    return [r["benefit"] for r in rows if r["model"] == model_name and r["frame"] == frame
            and r["type"] == ptype and r["source"] in sources]


def report(model_name, frame, a, b, sources):
    c = paired_compare(benefits(model_name, frame, a, sources), benefits(model_name, frame, b, sources))
    return (f"{a} - {b}: n={c['n']:<2} median={c['median']:+.2f} [95% CI {c['ci'][0]:+.2f}, "
            f"{c['ci'][1]:+.2f}]  r={c['r']:+.2f}  W={c['w']:.0f}  p={c['p']:.4f}")


for model_name in MODELS:
    print(f"\n{model_name}: joke-frame benefit (bits/token)")
    for frame_name, *_ in FRAME_PAIRS:
        print(f"--- frame: {frame_name}{' (primary)' if frame_name == 'reader' else ''} ---")
        for subset, sources in SUBSETS.items():
            medians = "  ".join(f"{t}={np.median(benefits(model_name, frame_name, t, sources)):+.2f}"
                                for t in TYPES)
            print(f"  {subset:<7} medians: {medians}")
            print(f"  {subset:<7} {report(model_name, frame_name, 'real', 'incongruous', sources)}")
            print(f"  {subset:<7} {report(model_name, frame_name, 'real', 'literal', sources)}")

with open(results_path("minimal_pairs_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "frame_pair", "name", "source", "punchline_type", "punchline", "n_tokens",
                "surprisal_no_frame", "surprisal_joke_frame", "surprisal_memo_frame", "joke_frame_benefit"])
    for r in rows:
        w.writerow([r["model"], r["frame"], r["name"], r["source"], r["type"], r["punchline"],
                    r["n_tokens"], f"{r['s_base']:.3f}", f"{r['s_joke']:.3f}", f"{r['s_memo']:.3f}",
                    f"{r['benefit']:.3f}"])

# Paired slope plot: each line is one joke, across its three punchlines.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, source, title in zip(axes, ("famous", "novel"), ("Famous jokes (n = 27)", "Novel jokes (n = 12)")):
    per_type = [benefits("google/gemma-2-2b-it", "reader", t, (source,)) for t in TYPES]
    for item in zip(*per_type):
        ax.plot(range(3), item, color="#8d99ae", alpha=0.45, linewidth=1, marker="o", markersize=3.5)
    color = COLORS[f"{source}_joke"]
    ax.plot(range(3), [np.median(v) for v in per_type], color=color, linewidth=3, marker="o",
            markersize=8, label="median")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Real punchline", "Incongruous foil", "Literal foil"])
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
axes[0].set_ylabel("Joke-frame benefit (bits/token)")
fig.suptitle('gemma-2-2b-it, "You are about to read a joke.": does the frame favour the real punchline '
             'over a matched foil?', fontsize=10)
save(fig, "fig_minimal_pairs.png")
