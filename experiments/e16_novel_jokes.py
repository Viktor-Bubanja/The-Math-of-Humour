"""
E16: Is the joke-frame benefit just memorisation?
==================================================
E14's benefit was strongest with the "well-known joke" wording, which invites
the model to recall famous jokes verbatim from training data. This reruns the
frame test on 12 jokes with new surface forms (classic mechanisms, new
wording), so recall can't predict the punchline. If the benefit survives on
them, it's genre conditioning, not memory.

Both frame wordings, both models. Effect sizes as Cliff's delta with
bootstrap 95% CIs.

Outputs: results/novel_jokes_results.csv, fig_novel_jokes.png
"""

import csv

import matplotlib.pyplot as plt

from humour.corpus import CONTROLS, FAMOUS_JOKES, NOVEL_JOKES
from humour.plotting import results_path, save, strip_plot
from humour.stats import compare, describe
from humour.surprisal import load, punchline_surprisal

MODELS = ["distilgpt2", "google/gemma-2-2b-it"]
FRAMES = {
    "wellknown": ("Here is a well-known joke: ", "Here is a routine factual note: "),
    "neutral": ("The following text is a joke.\n", "The following text is a factual statement.\n"),
}
CATS = ("novel_joke", "famous_joke", "control")
CORPUS = ([(n, "novel_joke", s, p) for n, _, s, p in NOVEL_JOKES]
          + [(n, "famous_joke", s, p) for n, _, s, p in FAMOUS_JOKES]
          + CONTROLS)

rows = []
for model_name in MODELS:
    print(f"\n=== {model_name} ===", flush=True)
    model, tokenizer = load(model_name)
    for frame_name, (frame_joke, frame_memo) in FRAMES.items():
        benefits = {cat: [] for cat in CATS}
        for name, cat, setup, punch in CORPUS:
            s_joke = punchline_surprisal(model, tokenizer, frame_joke, setup, punch)
            s_memo = punchline_surprisal(model, tokenizer, frame_memo, setup, punch)
            benefits[cat].append(s_memo - s_joke)
            rows.append(dict(model=model_name, frame=frame_name, name=name, category=cat,
                             s_joke=s_joke, s_memo=s_memo, benefit=s_memo - s_joke))

        print(f"--- frame: {frame_name} ---")
        for cat in CATS:
            print(f"  {cat:<12} {describe(benefits[cat])}")
        for cat in ("novel_joke", "famous_joke"):
            c = compare(benefits[cat], benefits["control"], ci=True)
            print(f"  {cat} vs control: U={c['u']:.1f}  p={c['p']:.4f}  Cliff's delta={c['delta']:+.2f} "
                  f"[95% CI {c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]")

with open(results_path("novel_jokes_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "frame", "name", "category", "surprisal_joke_frame",
                "surprisal_memo_frame", "joke_frame_benefit"])
    for r in rows:
        w.writerow([r["model"], r["frame"], r["name"], r["category"], f"{r['s_joke']:.3f}",
                    f"{r['s_memo']:.3f}", f"{r['benefit']:.3f}"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, frame_name, title in zip(axes, FRAMES, ('"Here is a well-known joke:"', '"The following text is a joke."')):
    groups = {cat: [r["benefit"] for r in rows
                    if "gemma" in r["model"] and r["frame"] == frame_name and r["category"] == cat]
              for cat in CATS}
    strip_plot(ax, groups, labels=["Novel jokes", "Famous jokes", "Prose controls"])
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("Joke-frame benefit (bits/token)")
fig.suptitle("gemma-2-2b-it: does the joke-frame benefit survive on jokes the model has never seen?", fontsize=10)
save(fig, "fig_novel_jokes.png")
