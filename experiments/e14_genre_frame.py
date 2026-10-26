"""
E14 / E15: Does telling the model "this is a joke" make punchlines cheaper?
============================================================================
E11-E13 found no trace of the setup misleading the model. This tests the other
half of the theory: if "joke" works as a latent interpretation, declaring it
should make joke punchlines cheaper to encode and should not help prose.

Every text gets the same two prefixes, so no per-item wording can leak hints:
    benefit = S(punch | memo frame + setup) - S(punch | joke frame + setup)
where S is mean punchline surprisal in bits/token. The test is the interaction:
is the benefit larger for jokes than for prose controls?

Usage:
    python experiments/e14_genre_frame.py            # E14: "Here is a well-known joke: "
    python experiments/e14_genre_frame.py neutral    # E15: "The following text is a joke.\\n"
Models: distilgpt2 and gemma-2-2b-it.
Outputs: results/genre_frame_results[_robustness].csv, fig_genre_frame[_robustness].png
"""

import csv
import sys

import matplotlib.pyplot as plt

from humour.corpus import ITEMS
from humour.plotting import results_path, save, strip_plot
from humour.stats import compare, describe
from humour.surprisal import load, punchline_surprisal

FRAMES = {
    "wellknown": ("Here is a well-known joke: ", "Here is a routine factual note: ", ""),
    "neutral": ("The following text is a joke.\n", "The following text is a factual statement.\n", "_robustness"),
}
FRAME_JOKE, FRAME_MEMO, SUFFIX = FRAMES[sys.argv[1] if len(sys.argv) > 1 else "wellknown"]
MODELS = ["distilgpt2", "google/gemma-2-2b-it"]
CATS = ("misdirection", "pun", "control")

all_results = {}
for model_name in MODELS:
    print(f"\n=== {model_name} ===", flush=True)
    model, tokenizer = load(model_name)
    results = []
    for name, cat, setup, punch in ITEMS:
        s_base = punchline_surprisal(model, tokenizer, "", setup, punch)
        s_joke = punchline_surprisal(model, tokenizer, FRAME_JOKE, setup, punch)
        s_memo = punchline_surprisal(model, tokenizer, FRAME_MEMO, setup, punch)
        results.append(dict(name=name, category=cat, s_base=s_base, s_joke=s_joke,
                            s_memo=s_memo, benefit=s_memo - s_joke))
        print(f"  {name:<16} {cat:<13} base={s_base:5.2f}  joke={s_joke:5.2f}  "
              f"memo={s_memo:5.2f}  benefit={s_memo - s_joke:+5.2f}", flush=True)
    all_results[model_name] = results


def vals(results, cat):
    return [r["benefit"] for r in results if r["category"] == cat]


for model_name, results in all_results.items():
    print(f"\n{model_name}: joke-frame benefit (bits/token)")
    for cat in CATS:
        print(f"  {cat:<13} {describe(vals(results, cat))}")
    c = compare(vals(results, "misdirection") + vals(results, "pun"), vals(results, "control"))
    print(f"  all jokes vs controls: U={c['u']:.1f}  p={c['p']:.4f}  delta={c['delta']:+.2f}")
    for a, b in (("misdirection", "control"), ("pun", "control"), ("misdirection", "pun")):
        c = compare(vals(results, a), vals(results, b))
        print(f"  {a:>13} vs {b:<13} U={c['u']:6.1f}  p={c['p']:.4f}")

with open(results_path(f"genre_frame_results{SUFFIX}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "name", "category", "surprisal_base", "surprisal_joke_frame",
                "surprisal_memo_frame", "joke_frame_benefit"])
    for model_name, results in all_results.items():
        for r in results:
            w.writerow([model_name, r["name"], r["category"], f"{r['s_base']:.2f}",
                        f"{r['s_joke']:.2f}", f"{r['s_memo']:.2f}", f"{r['benefit']:.2f}"])

fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.5))
for ax, model_name in zip(axes, MODELS):
    strip_plot(ax, {cat: vals(all_results[model_name], cat) for cat in CATS})
    ax.set_title(model_name.split("/")[-1], fontsize=10)
axes[0].set_ylabel("Joke-frame benefit (bits/token)\npunchline surprisal: memo frame − joke frame")
fig.suptitle('Does telling the model "this is a joke" make punchlines cheaper?', fontsize=10)
save(fig, f"fig_genre_frame{SUFFIX}.png")
