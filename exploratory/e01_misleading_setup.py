"""
Experiment: Does a joke's setup actively mislead the language model?

The blog predicts that good jokes have PPL(punchline | setup) > PPL(punchline alone),
because the setup primes the wrong model M1 — making the punchline MORE surprising
than it would be in isolation.

Misleading ratio = PPL(punchline | setup) / PPL(punchline alone)
  > 1 => setup successfully primed wrong expectations (misleading)
  ~ 1 => setup neutral, no priming effect
  < 1 => setup actually helped predict the punchline (coherent prose)
"""

import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()


def token_nll(context_text: str, target_text: str) -> float:
    """
    Compute the average NLL of target_text tokens conditioned on context_text.

    Both context and target are fed into the model together so that the
    target tokens attend to the context. Only the loss over target tokens
    is averaged — context tokens are excluded from the result.
    """
    context_ids = tokenizer.encode(context_text, return_tensors="pt")
    target_ids = tokenizer.encode(target_text, return_tensors="pt")

    n_context = context_ids.shape[1]
    n_target = target_ids.shape[1]

    # Concatenate: [context | target]
    full_ids = torch.cat([context_ids, target_ids], dim=1)

    with torch.no_grad():
        outputs = model(full_ids)

    logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    # The model predicts token[i+1] from logits[i].
    # We want NLL only for target tokens (positions n_context .. n_context+n_target-1).
    # logits for predicting position j come from logits[:, j-1, :].
    # Target token positions in full_ids: n_context to n_context+n_target-1 (inclusive).
    # So the logit slice we need: logits[:, n_context-1 : n_context+n_target-1, :]
    # And the label slice:        full_ids[:, n_context : n_context+n_target]

    logit_slice = logits[:, n_context - 1 : n_context + n_target - 1, :]
    label_slice = full_ids[:, n_context : n_context + n_target]

    log_probs = torch.nn.functional.log_softmax(logit_slice, dim=-1)
    token_log_probs = log_probs.gather(2, label_slice.unsqueeze(-1)).squeeze(-1)

    nll = -token_log_probs.mean().item()
    return nll


def perplexity_conditional(context: str, target: str) -> float:
    """PPL of target given context."""
    return math.exp(token_nll(context, target))


def perplexity_unconditional(target: str) -> float:
    """PPL of target with no context (empty string treated as single BOS token)."""
    # Feed target alone — the model still has its implicit positional prior
    ids = tokenizer.encode(target, return_tensors="pt")
    with torch.no_grad():
        outputs = model(ids)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    nll = -token_log_probs.mean().item()
    return math.exp(nll)


# ---------------------------------------------------------------------------
# Jokes and control to test
# ---------------------------------------------------------------------------

entries = [
    {
        "name": "Bungee jump",
        "setup": "ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR:",
        "punchline": "Don't lick my lips again.",
        "note": "Pronoun swap misdirection — 'my lips' shifts ownership",
    },
    {
        "name": "Atoms",
        "setup": "Why don't scientists trust atoms?",
        "punchline": "Because they make up everything.",
        "note": "Double meaning: 'make up' (fabricate vs. compose)",
    },
    {
        "name": "Skeleton",
        "setup": "Why did the skeleton go to the party alone?",
        "punchline": "Because he had no body to go with.",
        "note": "Homophone: 'nobody' vs. 'no body'",
    },
    {
        "name": "Time flies",
        "setup": "Time flies like an arrow.",
        "punchline": "Fruit flies like a banana.",
        "note": "Syntactic garden-path: 'flies' and 'like' reparse completely",
    },
    {
        "name": "Non-joke control",
        "setup": "The meeting is scheduled for Tuesday morning.",
        "punchline": "Please confirm your attendance by Friday.",
        "note": "Coherent prose — setup should NOT mislead",
    },
]

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

results = []
for entry in entries:
    ppl_alone = perplexity_unconditional(entry["punchline"])
    ppl_given = perplexity_conditional(entry["setup"], entry["punchline"])
    ratio = ppl_given / ppl_alone
    results.append(
        {
            "name": entry["name"],
            "ppl_alone": ppl_alone,
            "ppl_given": ppl_given,
            "ratio": ratio,
            "note": entry["note"],
        }
    )

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

name_w = 20
ppl_w = 16
ratio_w = 10

header = (
    f"{'Joke':<{name_w}} "
    f"{'PPL(punchline)':>{ppl_w}} "
    f"{'PPL(punchline|setup)':>{ppl_w}} "
    f"{'Ratio':>{ratio_w}}  "
    f"Note"
)
separator = "-" * (name_w + ppl_w * 2 + ratio_w + 20)

print()
print("=" * len(separator))
print("EXPERIMENT: Does a joke setup mislead the model into wrong expectations?")
print("=" * len(separator))
print()
print(header)
print(separator)

for r in results:
    verdict = "MISLEADING" if r["ratio"] > 1.05 else ("neutral" if r["ratio"] > 0.95 else "helpful")
    print(
        f"{r['name']:<{name_w}} "
        f"{r['ppl_alone']:>{ppl_w}.1f} "
        f"{r['ppl_given']:>{ppl_w}.1f} "
        f"{r['ratio']:>{ratio_w}.3f}x  "
        f"{r['note']}"
    )

print(separator)
print()
print("INTERPRETATION")
print("  Ratio > 1  =>  setup MISLEADS model (punchline more surprising with context)")
print("  Ratio ~ 1  =>  setup is neutral (no priming effect)")
print("  Ratio < 1  =>  setup HELPS model (coherent, predictable continuation)")
print()

# Highlight the control comparison
control = next(r for r in results if r["name"] == "Non-joke control")
jokes_only = [r for r in results if r["name"] != "Non-joke control"]
avg_joke_ratio = sum(r["ratio"] for r in jokes_only) / len(jokes_only)

print(f"  Non-joke control ratio : {control['ratio']:.3f}x")
print(f"  Average joke ratio     : {avg_joke_ratio:.3f}x")
print()
if avg_joke_ratio > control["ratio"]:
    print("  => Jokes show HIGHER misleading ratios than the control, supporting")
    print("     the hypothesis that good setups prime the WRONG model M1.")
else:
    print("  => Jokes do NOT consistently outperform the control in misleading ratio.")
    print("     The hypothesis is not supported for this model / joke sample.")
print()
