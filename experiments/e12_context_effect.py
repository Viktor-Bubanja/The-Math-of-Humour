"""
E12 / E13: Does the setup actively mislead?
============================================
E11 showed spike magnitude mostly measures lexical rarity. Misdirection theory
makes a sharper prediction: the setup builds a wrong model M1, so for the twist
token the setup should be WORSE than no context at all.

For each punchline token t:
    delta_t = S(t | BOS + setup + punch_<t) - S(t | BOS + punch_<t)

Positive delta = the setup made t harder to predict. Per item we keep the max
delta over the punchline ("worst-case context damage") and the mean.
Result: no separation between jokes and prose at any scale (see README).

Usage:
    python experiments/e12_context_effect.py                        # E12: distilgpt2, gpt2-large
    python experiments/e12_context_effect.py google/gemma-2-2b-it   # E13
Outputs: results/context_effect_results[_<model>].csv, fig_context_effect[_<model>].png
"""

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

from humour.corpus import ITEMS
from humour.plotting import results_path, save, strip_plot
from humour.stats import compare, describe
from humour.surprisal import bos_ids, continuation_surprisals, load

MODELS = sys.argv[1:] or ["distilgpt2", "gpt2-large"]
# Non-default runs get a suffix so they don't overwrite the default outputs.
SUFFIX = "" if not sys.argv[1:] else "_" + "_".join(m.split("/")[-1] for m in MODELS)
CATS = ("misdirection", "pun", "control")

all_results = {}
for model_name in MODELS:
    print(f"\n=== {model_name} ===", flush=True)
    model, tokenizer = load(model_name)
    bos = bos_ids(tokenizer)

    results = []
    for name, cat, setup, punch in ITEMS:
        setup_ids = tokenizer(setup, add_special_tokens=False)["input_ids"]
        punch_ids = tokenizer(punch, add_special_tokens=False)["input_ids"]
        with_ctx = continuation_surprisals(model, bos + setup_ids, punch_ids)
        no_ctx = continuation_surprisals(model, bos, punch_ids)
        deltas = [a - b for a, b in zip(with_ctx, no_ctx)]

        i = int(np.argmax(deltas))
        results.append(dict(name=name, category=cat, max_delta=deltas[i],
                            max_delta_token=tokenizer.decode([punch_ids[i]]),
                            mean_delta=float(np.mean(deltas)),
                            frac_positive=float(np.mean([d > 0 for d in deltas]))))
        print(f"  {name:<16} {cat:<13} max_delta={deltas[i]:+6.2f} on "
              f"{tokenizer.decode([punch_ids[i]]).strip()!r:<14} mean={np.mean(deltas):+6.2f}", flush=True)
    all_results[model_name] = results


def vals(results, cat, key):
    return [r[key] for r in results if r["category"] == cat]


for model_name, results in all_results.items():
    print(f"\n{model_name}")
    for key in ("max_delta", "mean_delta"):
        print(f"  {key}:")
        for cat in CATS:
            print(f"    {cat:<13} {describe(vals(results, cat, key))}")
    print("  Mann-Whitney U (two-sided), max_delta:")
    for a, b in (("misdirection", "pun"), ("misdirection", "control"), ("pun", "control")):
        c = compare(vals(results, a, "max_delta"), vals(results, b, "max_delta"))
        print(f"    {a:>13} vs {b:<13} U={c['u']:6.1f}  p={c['p']:.4f}")

with open(results_path(f"context_effect_results{SUFFIX}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "name", "category", "max_delta", "max_delta_token", "mean_delta", "frac_positive"])
    for model_name, results in all_results.items():
        for r in results:
            w.writerow([model_name, r["name"], r["category"], f"{r['max_delta']:.2f}",
                        r["max_delta_token"].strip(), f"{r['mean_delta']:.2f}", f"{r['frac_positive']:.2f}"])

fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.5), sharey=True)
for ax, model_name in zip(np.atleast_1d(axes), MODELS):
    strip_plot(ax, {cat: vals(all_results[model_name], cat, "max_delta") for cat in CATS})
    ax.set_title(model_name, fontsize=10)
np.atleast_1d(axes)[0].set_ylabel("Max context damage (bits)\nsurprisal with setup − without")
fig.suptitle("Does the setup actively mislead? Positive = setup makes a punchline token HARDER to predict",
             fontsize=10)
save(fig, f"fig_context_effect{SUFFIX}.png")
