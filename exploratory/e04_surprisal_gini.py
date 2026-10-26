"""
Experiment: Do jokes concentrate surprisal at a spike (high Gini) vs. prose (low Gini)?

Hypothesis: The punchline/twist in a joke creates a single high-surprisal token,
making the distribution of per-token surprisal more unequal (higher Gini coefficient)
than factual prose of similar length.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def token_surprisals(text: str, model, tokenizer) -> np.ndarray:
    """
    Returns per-token surprisal: -log2 p(token_t | token_0 ... token_{t-1})
    for each token in the text (excluding the first token, which has no context).
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]  # shape: (1, seq_len)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits

    # Shift so that token i predicts token i+1
    # logits[0, :-1, :] predicts input_ids[0, 1:]
    shift_logits = logits[0, :-1, :]        # (seq_len-1, vocab_size)
    shift_labels = input_ids[0, 1:]          # (seq_len-1,)

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    # Gather the log-prob of each actual next token
    token_log_probs = log_probs[torch.arange(len(shift_labels)), shift_labels]

    # Surprisal = -log2 p  (convert from natural log)
    surprisals = (-token_log_probs / torch.log(torch.tensor(2.0))).numpy()
    return surprisals


def gini(arr: np.ndarray) -> float:
    """
    Standard Gini coefficient for a non-negative array.
    Gini = 0: perfectly uniform; Gini = 1: all mass on one element.
    Formula: (2 * sum(rank * value) / (n * sum(values))) - (n+1)/n
    where rank is 1-indexed after sorting ascending.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.sum() == 0:
        return 0.0
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    ranks = np.arange(1, n + 1)  # 1-indexed
    return (2.0 * np.sum(ranks * arr_sorted) / (n * arr_sorted.sum())) - (n + 1) / n


def analyze(text: str, label: str, model, tokenizer) -> dict:
    surprisals = token_surprisals(text, model, tokenizer)
    mean_s = float(np.mean(surprisals))
    max_s = float(np.max(surprisals))
    g = gini(surprisals)
    peak_ratio = max_s / mean_s if mean_s > 0 else float("inf")

    print(f"\n[{label}]")
    print(f"  Text     : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Tokens   : {len(surprisals) + 1}")
    print(f"  Mean surprisal : {mean_s:.3f} bits")
    print(f"  Max surprisal  : {max_s:.3f} bits")
    print(f"  Gini           : {g:.4f}")
    print(f"  Max/Mean ratio : {peak_ratio:.3f}")

    return {"label": label, "text": text, "mean": mean_s, "max": max_s,
            "gini": g, "peak_ratio": peak_ratio}


def main():
    print("Loading distilgpt2 model and tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()
    print("Model loaded.\n")

    jokes = [
        "ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR: Don't lick my lips again.",
        "Why don't scientists trust atoms? Because they make up everything.",
        "Time flies like an arrow. Fruit flies like a banana.",
        "I asked the librarian if they had books about paranoia. She whispered: they're right behind you.",
    ]

    prose = [
        "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow.",
        "Scientists have discovered that regular exercise can significantly reduce the risk of cardiovascular disease and improve overall mental health.",
        "The meeting was scheduled for Tuesday morning. All department heads were asked to confirm their attendance by Friday afternoon.",
        "Paris is the capital city of France and is known for its art, culture, and the iconic Eiffel Tower structure.",
    ]

    print("=" * 70)
    print("JOKES")
    print("=" * 70)
    joke_results = [analyze(t, f"Joke {i+1}", model, tokenizer)
                    for i, t in enumerate(jokes)]

    print("\n" + "=" * 70)
    print("PROSE")
    print("=" * 70)
    prose_results = [analyze(t, f"Prose {i+1}", model, tokenizer)
                     for i, t in enumerate(prose)]

    # Summary
    avg_joke_gini = np.mean([r["gini"] for r in joke_results])
    avg_prose_gini = np.mean([r["gini"] for r in prose_results])
    avg_joke_peak = np.mean([r["peak_ratio"] for r in joke_results])
    avg_prose_peak = np.mean([r["peak_ratio"] for r in prose_results])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Average Gini  — Jokes: {avg_joke_gini:.4f}   Prose: {avg_prose_gini:.4f}")
    print(f"  Average Max/Mean — Jokes: {avg_joke_peak:.3f}   Prose: {avg_prose_peak:.3f}")

    hypothesis_holds = avg_joke_gini > avg_prose_gini
    direction = "HIGHER" if hypothesis_holds else "LOWER"
    print(f"\n  Hypothesis: jokes have higher Gini than prose.")
    print(f"  Result: joke Gini is {direction} than prose Gini "
          f"({avg_joke_gini:.4f} vs {avg_prose_gini:.4f}).")
    print(f"  => Hypothesis {'SUPPORTED' if hypothesis_holds else 'NOT SUPPORTED'} "
          f"by this experiment.")


if __name__ == "__main__":
    main()
