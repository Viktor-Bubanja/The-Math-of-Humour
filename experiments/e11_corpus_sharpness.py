"""
E11: Does a joke's surprisal profile look different from prose?
================================================================
Scales the early five-text "resolution sharpness" result (exploratory/e06) to
the 37-text corpus: 12 misdirection jokes, 15 puns, 10 prose controls.
Categories and punchline boundaries were fixed before running any model.

Metrics per item, over the punchline:
  - peak surprisal (bits)
  - resolution sharpness = peak / mean(post-peak punchline tokens)
  - tokens-to-recover = post-peak tokens until surprisal < setup mean

Test: two-sided Mann-Whitney U between category pairs.
Result: no separation on any metric (see README).

Model: distilgpt2.
Outputs: results/corpus_sharpness_results.csv, fig_sharpness_by_category.png,
         fig_surprisal_cliff.png (token profiles for Time flies + Bungee).
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from humour.corpus import ITEMS
from humour.plotting import results_path, save, strip_plot
from humour.stats import compare
from humour.surprisal import load, token_surprisals

MODEL_NAME = "distilgpt2"
CATS = ("misdirection", "pun", "control")

model, tokenizer = load(MODEL_NAME)


def analyze(name: str, category: str, setup: str, punchline: str) -> dict:
    setup_ids = tokenizer(setup, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(setup + punchline, add_special_tokens=False)["input_ids"]
    n_setup = len(setup_ids)
    # BPE can merge across the boundary; if it did, "punchline tokens" would silently shift.
    assert full_ids[:n_setup] == setup_ids, f"[{name}] tokenization mismatch"

    surp = token_surprisals(model, tokenizer, setup + punchline)
    setup_vals = [s for _, s in surp[1:n_setup]]  # skip contextless token 0
    punch = surp[n_setup:]
    punch_vals = [s for _, s in punch]

    peak_idx = int(np.argmax(punch_vals))
    peak = punch_vals[peak_idx]
    post = punch_vals[peak_idx + 1:]
    pre_peak_mean = float(np.mean(setup_vals))
    post_mean = float(np.mean(post)) if post else float("nan")
    sharpness = peak / post_mean if post else float("nan")
    tokens_to_recover = next((i + 1 for i, v in enumerate(post) if v < pre_peak_mean), None)

    return dict(name=name, category=category, n_setup=n_setup, surprisals=surp,
                pre_peak_mean=pre_peak_mean, peak=peak, peak_token=punch[peak_idx][0],
                peak_idx=peak_idx, post_mean=post_mean, sharpness=sharpness,
                tokens_to_recover=tokens_to_recover)


results = [analyze(*item) for item in ITEMS]


def vals(cat: str, key: str) -> list[float]:
    return [r[key] for r in results if r["category"] == cat and not np.isnan(r[key])]


with open(results_path("corpus_sharpness_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "category", "peak_bits", "peak_token", "pre_peak_mean",
                "post_peak_mean", "sharpness", "tokens_to_recover"])
    for r in results:
        w.writerow([r["name"], r["category"], f"{r['peak']:.2f}", r["peak_token"].strip(),
                    f"{r['pre_peak_mean']:.2f}",
                    "n/a" if np.isnan(r["post_mean"]) else f"{r['post_mean']:.2f}",
                    "n/a" if np.isnan(r["sharpness"]) else f"{r['sharpness']:.2f}",
                    r["tokens_to_recover"] if r["tokens_to_recover"] is not None else "never"])

print(f"{'Item':<16} {'Cat':<13} {'Peak':>6} {'Token':<14} {'Sharp':>7} {'Recov':>6}")
for r in sorted(results, key=lambda r: (r["category"], -r["peak"])):
    sh = "n/a" if np.isnan(r["sharpness"]) else f"{r['sharpness']:.2f}"
    print(f"{r['name']:<16} {r['category']:<13} {r['peak']:>6.2f} {r['peak_token'].strip():<14} "
          f"{sh:>7} {str(r['tokens_to_recover'] or 'never'):>6}")

print("\nTokens to recover:")
for cat in CATS:
    recs = [r["tokens_to_recover"] for r in results if r["category"] == cat]
    print(f"  {cat:<13} {sum(1 for x in recs if x == 1)}/{len(recs)} recover in exactly 1 token")

print("\nMann-Whitney U (two-sided):")
for key in ("peak", "sharpness"):
    for a, b in (("misdirection", "pun"), ("misdirection", "control"), ("pun", "control")):
        c = compare(vals(a, key), vals(b, key))
        print(f"  {key:<10} {a:>13} vs {b:<13} U={c['u']:6.1f}  p={c['p']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, key, label in zip(axes, ("peak", "sharpness"),
                          ("Peak punchline surprisal (bits)", "Resolution sharpness (peak / post-peak mean)")):
    strip_plot(ax, {cat: vals(cat, key) for cat in CATS}, zero_line=False)
    ax.set_ylabel(label)
fig.suptitle(f"37-text corpus, {MODEL_NAME} — bars mark category medians", fontsize=10)
save(fig, "fig_sharpness_by_category.png")

fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
for ax, name in zip(axes, ("Time flies", "Bungee")):
    r = next(r for r in results if r["name"] == name)
    toks = [t for t, _ in r["surprisals"]]
    ax.plot(range(len(toks)), [v for _, v in r["surprisals"]],
            marker="o", markersize=4, linewidth=1.3, color="#2b2d42")
    ax.axvline(r["n_setup"] - 0.5, color="#8d99ae", linestyle="--", linewidth=1)
    peak_x = r["n_setup"] + r["peak_idx"]
    ax.annotate(f"peak: {r['peak']:.1f} bits", (peak_x, r["peak"]),
                textcoords="offset points", xytext=(8, -2), fontsize=9, color="#d1495b")
    ax.scatter([peak_x], [r["peak"]], color="#d1495b", zorder=5, s=60)
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels([t.replace(" ", "·") for t in toks], rotation=60, fontsize=7, ha="right")
    ax.set_ylabel("Surprisal (bits)")
    ax.set_title(name, fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Token-by-token surprisal ({MODEL_NAME}); dashed line = setup/punchline boundary", fontsize=10)
save(fig, "fig_surprisal_cliff.png")
