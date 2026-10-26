"""
Experiment: Token-level surprisal profiling of jokes.

The question: Do jokes concentrate surprisal at a specific "twist word" in the punchline?
We compute -log2 p(token_t | token_0...token_{t-1}) for every token using distilgpt2,
then inspect the punchline portion for peak surprisal.
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

MODEL_NAME = "distilgpt2"
print(f"Loading {MODEL_NAME}...", flush=True)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Jokes: (full_text, setup_text, label)
# ---------------------------------------------------------------------------

jokes = [
    (
        "ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR: Don't lick my lips again.",
        "ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR:",
        "Bungee jump",
    ),
    (
        "Why don't scientists trust atoms? Because they make up everything.",
        "Why don't scientists trust atoms?",
        "Atoms",
    ),
    (
        "Time flies like an arrow. Fruit flies like a banana.",
        "Time flies like an arrow.",
        "Time flies",
    ),
]

# ---------------------------------------------------------------------------
# Surprisal computation
# ---------------------------------------------------------------------------

def compute_surprisals(text: str) -> tuple[list[str], list[float]]:
    """Return (token_strings, surprisals_in_bits) for every token in text."""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]  # shape (1, T)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # logits shape: (1, T, vocab_size)
        logits = outputs.logits[0]  # (T, vocab_size)

    # Token t's surprisal = -log2 p(token_t | token_0..token_{t-1})
    # logits[t-1] gives the distribution *before* seeing token t, predicting token t.
    # So for token at position t (t >= 1):  surprisal[t] from logits[t-1].
    # Token 0 has no prior context — assign NaN or 0.

    log_probs = torch.log_softmax(logits, dim=-1)  # (T, vocab)
    T = input_ids.shape[1]

    surprisals = []
    token_ids = input_ids[0].tolist()

    # Position 0: undefined (no prior context), use 0.0
    surprisals.append(0.0)
    for t in range(1, T):
        lp = log_probs[t - 1, token_ids[t]].item()  # log p(token_t | ..token_{t-1})
        surprisals.append(-lp / np.log(2))  # convert nats to bits

    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    return token_strings, surprisals


def tokenize_text(text: str) -> list[int]:
    return tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()


# ---------------------------------------------------------------------------
# ASCII bar chart
# ---------------------------------------------------------------------------

def bar_chart(labels: list[str], values: list[float], width: int = 40) -> str:
    if not values:
        return "(no tokens)"
    max_val = max(values) if max(values) > 0 else 1.0
    lines = []
    for lbl, val in zip(labels, values):
        bar_len = int(round(val / max_val * width))
        bar = "#" * bar_len
        lines.append(f"  {lbl:>18s} | {bar:<{width}s} {val:.2f} bits")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

for full_text, setup_text, label in jokes:
    print("=" * 70)
    print(f"JOKE: {label}")
    print(f"Full text: {full_text}")
    print()

    # Tokenize full text and setup separately
    full_ids = tokenize_text(full_text)
    setup_ids = tokenize_text(setup_text)
    n_setup = len(setup_ids)

    # Compute surprisals over the full string (so punchline tokens get proper context)
    token_strings, surprisals = compute_surprisals(full_text)

    # Split
    setup_tokens = token_strings[:n_setup]
    setup_surps = surprisals[:n_setup]
    punch_tokens = token_strings[n_setup:]
    punch_surps = surprisals[n_setup:]

    if not punch_tokens:
        print("  (no punchline tokens found — check split)\n")
        continue

    # Decode tokens for display (replace Ġ with space prefix marker)
    def display(tok: str) -> str:
        return tok.replace("Ġ", "_").replace("Ċ", "\\n")

    # Stats
    punch_display = [display(t) for t in punch_tokens]
    max_idx = int(np.argmax(punch_surps))
    top3_idx = sorted(range(len(punch_surps)), key=lambda i: punch_surps[i], reverse=True)[:3]

    print(f"Setup tokens  ({n_setup}): {' '.join(display(t) for t in setup_tokens)}")
    print(f"Punchline tokens ({len(punch_tokens)}): {' '.join(punch_display)}")
    print()

    # Mean surprisals
    mean_setup = np.mean(setup_surps[1:]) if len(setup_surps) > 1 else 0.0
    mean_punch = np.mean(punch_surps) if punch_surps else 0.0
    print(f"Mean surprisal — setup: {mean_setup:.2f} bits | punchline: {mean_punch:.2f} bits")
    print()

    # Peak token
    peak_tok = punch_tokens[max_idx]
    peak_decoded = tokenizer.decode([tokenizer.convert_tokens_to_ids(peak_tok)]).strip()
    print(f"Peak surprisal token in punchline:")
    print(f"  Token #{max_idx} → '{display(peak_tok)}' (decoded: '{peak_decoded}')  {punch_surps[max_idx]:.2f} bits")
    print()

    # Top-3
    print("Top-3 most surprising punchline tokens:")
    for rank, i in enumerate(top3_idx, 1):
        decoded = tokenizer.decode([tokenizer.convert_tokens_to_ids(punch_tokens[i])]).strip()
        print(f"  #{rank}: '{display(punch_tokens[i])}' (decoded: '{decoded}')  {punch_surps[i]:.2f} bits")
    print()

    # Bar chart
    print("Surprisal profile — punchline tokens (scaled to 40 chars):")
    print(bar_chart(punch_display, punch_surps, width=40))
    print()

print("=" * 70)
print("\nDISCUSSION")
print("-" * 70)
discussion = """
For each joke the script identifies the single token carrying peak surprisal
in the punchline and checks whether it aligns with the semantic "twist":

1. Bungee jump ("Don't lick my lips again.")
   The twist is that the INSTRUCTOR is the one whose lips were licked —
   the joke recontextualises whose lips "my" refers to.  We expect the
   peak to fall on "my" or "lips", the pronoun that reveals the victim.

2. Atoms ("Because they make up everything.")
   The twist is the double meaning of "make up" (lie vs. compose).
   The peak should fall on "everything" or "make", the pivot of the pun.

3. Time flies ("Fruit flies like a banana.")
   The twist is the garden-path reparse: "Fruit flies" becomes a noun
   phrase (a type of fly), "like" becomes a verb (prefer), and "a banana"
   is the object.  The peak should fall on "banana" or "flies", where the
   garden-path forces a reparse.

Information theory predicts that the twist word is surprising because the
language model (trained on non-jokey text) assigns low probability to it
given the setup context.  A high surprisal at the semantic pivot confirms
the joke compresses its humour into that single token.
"""
print(discussion)
