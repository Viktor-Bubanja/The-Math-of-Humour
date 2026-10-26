"""
Conditional Gini experiment: Is surprisal more concentrated within the
punchline of a joke compared to the second sentence of prose?

Each punchline/second-sentence is evaluated conditioned on its setup,
so we isolate only the punchline's token-level surprisal distribution.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def punchline_conditional_surprisals(setup_text, punchline_text, model, tokenizer):
    """
    Return per-token surprisal values (bits) for each punchline token,
    conditioned on the full setup as context.

    Technique: feed [setup + punchline] together, take logits, slice out
    only the punchline token positions.
    """
    # Encode setup and punchline separately to know the boundary
    setup_ids = tokenizer.encode(setup_text, add_special_tokens=False)
    # Encode punchline with a leading space so it joins naturally
    punchline_ids = tokenizer.encode(" " + punchline_text, add_special_tokens=False)

    full_ids = setup_ids + punchline_ids
    input_ids = torch.tensor([full_ids])

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    # For token at position i, the logit at position i-1 gives P(token_i | tokens_0..i-1)
    # We want surprisals for the punchline tokens, which start at index len(setup_ids)
    n_setup = len(setup_ids)
    n_punch = len(punchline_ids)

    surprisals = []
    for i in range(n_punch):
        # Position in the sequence for this punchline token
        pos = n_setup + i
        # The logit that predicts this token comes from position pos-1
        logit_pos = pos - 1
        logits_at_pos = logits[0, logit_pos, :]  # (vocab_size,)
        log_probs = torch.log_softmax(logits_at_pos, dim=-1)
        token_id = full_ids[pos]
        log_prob_nats = log_probs[token_id].item()
        # Convert from nats to bits
        surprisal_bits = -log_prob_nats / np.log(2)
        surprisals.append(surprisal_bits)

    return np.array(surprisals), punchline_ids, tokenizer


def gini(arr):
    """
    Standard Gini coefficient.
    arr: array of non-negative values.
    Returns a value in [0, 1]: 0 = perfectly equal, 1 = maximally concentrated.
    """
    arr = np.array(arr, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr_sorted = np.sort(arr)  # ascending
    n = len(arr_sorted)
    # gini = (2 * sum((i+1)*arr[i] for i in range(n)) / (n * sum(arr))) - (n+1)/n
    indices = np.arange(1, n + 1)
    gini_val = (2.0 * np.sum(indices * arr_sorted) / (n * arr_sorted.sum())) - (n + 1) / n
    return gini_val


def analyze_pair(label, setup_text, punchline_text, model, tokenizer):
    surprisals, punchline_ids, tok = punchline_conditional_surprisals(
        setup_text, punchline_text, model, tokenizer
    )
    n = len(surprisals)
    mean_s = surprisals.mean()
    max_s = surprisals.max()
    g = gini(surprisals)
    ratio = max_s / mean_s if mean_s > 0 else float("inf")
    peak_idx = int(np.argmax(surprisals))
    peak_token = tok.decode([punchline_ids[peak_idx]])

    print(f"  [{label}]")
    print(f"    N tokens     : {n}")
    print(f"    Mean surprisal: {mean_s:.3f} bits")
    print(f"    Max surprisal : {max_s:.3f} bits")
    print(f"    Gini          : {g:.4f}")
    print(f"    Max/Mean ratio: {ratio:.3f}")
    print(f"    Peak token    : '{peak_token}' (surprisal {surprisals[peak_idx]:.3f} bits)")
    print()

    return {"gini": g, "ratio": ratio, "mean": mean_s, "max": max_s, "n": n}


def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print(f"Model: {MODEL_NAME}\n")

    # ── JOKES ──────────────────────────────────────────────────────────────────
    jokes = [
        (
            "Bungee",
            "ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR:",
            "Don't lick my lips again.",
        ),
        (
            "Time flies",
            "Time flies like an arrow.",
            "Fruit flies like a banana.",
        ),
        (
            "Atoms",
            "Why don't scientists trust atoms?",
            "Because they make up everything.",
        ),
        (
            "Librarian",
            "I asked the librarian if they had books about paranoia. She whispered:",
            "they're right behind you.",
        ),
        (
            "Skeleton",
            "Why did the skeleton go to the party alone?",
            "Because he had no body to go with.",
        ),
    ]

    # ── PROSE CONTROLS ─────────────────────────────────────────────────────────
    prose = [
        (
            "Water cycle",
            "The water cycle describes how water evaporates from surfaces.",
            "It then rises into the atmosphere and condenses into rain or snow.",
        ),
        (
            "Exercise",
            "Scientists have discovered that regular exercise reduces cardiovascular risk.",
            "Mental health benefits have also been documented across multiple studies.",
        ),
        (
            "Meeting",
            "The meeting is scheduled for Tuesday morning.",
            "Please confirm your attendance by Friday.",
        ),
        (
            "Paris",
            "Paris is the capital city of France.",
            "It is known for the Eiffel Tower and world-class cuisine.",
        ),
        (
            "Earnings",
            "The company reported strong earnings in the third quarter.",
            "Revenue increased by twelve percent compared to the previous year.",
        ),
    ]

    joke_stats = []
    prose_stats = []

    print("=" * 60)
    print("JOKES (punchline conditioned on setup)")
    print("=" * 60)
    for label, setup, punchline in jokes:
        stats = analyze_pair(label, setup, punchline, model, tokenizer)
        joke_stats.append(stats)

    print("=" * 60)
    print("PROSE (second sentence conditioned on first)")
    print("=" * 60)
    for label, setup, punchline in prose:
        stats = analyze_pair(label, setup, punchline, model, tokenizer)
        prose_stats.append(stats)

    # ── SUMMARY ────────────────────────────────────────────────────────────────
    joke_ginis  = [s["gini"]  for s in joke_stats]
    prose_ginis = [s["gini"]  for s in prose_stats]
    joke_ratios  = [s["ratio"]  for s in joke_stats]
    prose_ratios = [s["ratio"]  for s in prose_stats]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Jokes  — mean Gini: {np.mean(joke_ginis):.4f}  |  mean max/mean: {np.mean(joke_ratios):.3f}")
    print(f"  Prose  — mean Gini: {np.mean(prose_ginis):.4f}  |  mean max/mean: {np.mean(prose_ratios):.3f}")
    print()

    gini_diff  = np.mean(joke_ginis)  - np.mean(prose_ginis)
    ratio_diff = np.mean(joke_ratios) - np.mean(prose_ratios)

    hypothesis_gini  = gini_diff  > 0
    hypothesis_ratio = ratio_diff > 0

    print("HYPOTHESIS: jokes are more concentrated than prose within punchline tokens.")
    print(f"  Gini  delta (jokes − prose): {gini_diff:+.4f}  → {'SUPPORTED' if hypothesis_gini  else 'NOT SUPPORTED'}")
    print(f"  Ratio delta (jokes − prose): {ratio_diff:+.3f}  → {'SUPPORTED' if hypothesis_ratio else 'NOT SUPPORTED'}")

    if hypothesis_gini and hypothesis_ratio:
        verdict = "SUPPORTED by both metrics."
    elif hypothesis_gini or hypothesis_ratio:
        verdict = "PARTIALLY SUPPORTED (one metric agrees, one does not)."
    else:
        verdict = "NOT SUPPORTED by either metric."

    print(f"\nOverall verdict: {verdict}")


if __name__ == "__main__":
    main()
