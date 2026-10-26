"""
Experiment: Expected vs Actual Punchline Surprisal
---------------------------------------------------
For each joke, measure how far off-manifold the actual punchline is
compared to what the model would have naturally generated (greedy decoding).

Metric: off_manifold_ratio = PPL(actual_punchline | setup) / PPL(expected_completion | setup)
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_NAME = "distilgpt2"
print(f"Loading {MODEL_NAME}...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.\n")

# ── Core functions ─────────────────────────────────────────────────────────────

def conditional_ppl(context: str, target: str, model, tokenizer) -> float:
    """
    Compute PPL of `target` conditioned on `context`.
    Feed [context + target] through the model, compute average NLL
    over target token positions only.
    """
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    target_ids  = tokenizer.encode(target,  add_special_tokens=False)

    if not target_ids:
        return float("nan")

    full_ids = context_ids + target_ids
    input_ids = torch.tensor([full_ids])

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.logits: [1, seq_len, vocab_size]
        logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Shifted: logit[i] predicts token[i+1]
    # We want NLL for positions context_len .. context_len+target_len-1
    # which correspond to logits at positions context_len-1 .. context_len+target_len-2
    context_len = len(context_ids)
    target_len  = len(target_ids)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # logits for predicting target tokens: indices [context_len-1 .. context_len+target_len-2]
    pred_logits = logits[context_len - 1 : context_len + target_len - 1]  # [target_len, vocab]
    target_tensor = torch.tensor(target_ids)                               # [target_len]

    nll_per_token = loss_fn(pred_logits, target_tensor)  # [target_len]
    mean_nll = nll_per_token.mean().item()
    return float(np.exp(mean_nll))


def generate_greedy(setup_text: str, n_tokens: int, model, tokenizer) -> str:
    """
    Greedy-decode exactly `n_tokens` new tokens beyond `setup_text`.
    Returns only the newly generated text (decoded).
    """
    input_ids = tokenizer.encode(setup_text, add_special_tokens=False, return_tensors="pt")
    n_context = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            min_new_tokens=n_tokens,
            do_sample=False,          # greedy
            pad_token_id=tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[0, n_context:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


# ── Jokes & control ────────────────────────────────────────────────────────────

items = [
    {
        "name": "Bungee",
        "setup": (
            "ME: [licking lips in anticipation] I'm nervous. "
            "I've never done a bungee jump before. INSTRUCTOR:"
        ),
        "punchline": "Don't lick my lips again.",
    },
    {
        "name": "Time flies",
        "setup": "Time flies like an arrow.",
        "punchline": "Fruit flies like a banana.",
    },
    {
        "name": "Atoms",
        "setup": "Why don't scientists trust atoms?",
        "punchline": "Because they make up everything.",
    },
    {
        "name": "Librarian",
        "setup": (
            "I asked the librarian if they had books about paranoia. "
            "She whispered:"
        ),
        "punchline": "they're right behind you.",
    },
    {
        "name": "Control (not a joke)",
        "setup": "The meeting is scheduled for Tuesday morning.",
        "punchline": "Please confirm your attendance by Friday.",
    },
]

# ── Run experiment ─────────────────────────────────────────────────────────────

DIVIDER = "=" * 70

results = []

for item in items:
    name      = item["name"]
    setup     = item["setup"]
    punchline = item["punchline"]

    n_tokens  = len(tokenizer.encode(punchline, add_special_tokens=False))
    expected  = generate_greedy(setup, n_tokens, model, tokenizer)

    ppl_expected = conditional_ppl(setup, expected,  model, tokenizer)
    ppl_actual   = conditional_ppl(setup, punchline, model, tokenizer)
    ratio        = ppl_actual / ppl_expected

    results.append({
        "name": name,
        "setup": setup,
        "expected": expected,
        "punchline": punchline,
        "ppl_expected": ppl_expected,
        "ppl_actual": ppl_actual,
        "ratio": ratio,
    })

    print(DIVIDER)
    print(f"[{name}]")
    print(f"  Setup    : {setup}")
    print(f"  Expected : {expected!r}")
    print(f"  Actual   : {punchline!r}")
    print(f"  PPL(expected|setup) = {ppl_expected:.2f}")
    print(f"  PPL(actual|setup)   = {ppl_actual:.2f}")
    print(f"  Off-manifold ratio  = {ratio:.2f}x")
    print()

# ── Summary ────────────────────────────────────────────────────────────────────

print(DIVIDER)
print("SUMMARY")
print(DIVIDER)
print(f"{'Item':<25} {'PPL(exp)':>10} {'PPL(act)':>10} {'Ratio':>8}")
print("-" * 55)
for r in results:
    flag = "  <-- control" if "Control" in r["name"] else ""
    print(f"{r['name']:<25} {r['ppl_expected']:>10.2f} {r['ppl_actual']:>10.2f} {r['ratio']:>8.2f}x{flag}")

joke_ratios    = [r["ratio"] for r in results if "Control" not in r["name"]]
control_ratios = [r["ratio"] for r in results if "Control"     in r["name"]]

print()
print(f"Average off-manifold ratio  — jokes  : {np.mean(joke_ratios):.2f}x")
print(f"Average off-manifold ratio  — control: {np.mean(control_ratios):.2f}x")
print()
if np.mean(joke_ratios) > np.mean(control_ratios):
    print("CONCLUSION: Jokes have HIGHER off-manifold ratios than the control,")
    print("  supporting the hypothesis that punchlines are surprisingly far from")
    print("  the model's expected manifold — a signature of humor.")
else:
    print("CONCLUSION: Jokes do NOT show higher off-manifold ratios than the control.")
    print("  (Possibly distilgpt2 cannot distinguish joke punchlines from mundane text.)")
