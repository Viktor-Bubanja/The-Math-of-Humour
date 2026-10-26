"""
Experiment: Does misleading ratio (PPL(punchline|setup) / PPL(punchline alone))
increase with model scale? At some scale, does it cross 1.0?

Misleading ratio > 1 means setup made punchline MORE surprising (setup successfully misleads).
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Jokes to test
# ---------------------------------------------------------------------------
jokes = [
    {
        "name": "Bungee",
        "setup": "ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR:",
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
        "name": "Control",
        "setup": "The meeting is scheduled for Tuesday morning.",
        "punchline": "Please confirm your attendance by Friday.",
    },
]

# ---------------------------------------------------------------------------
# Models in order of size; gpt2-large is attempted last
# ---------------------------------------------------------------------------
models_to_try = [
    "distilgpt2",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
]


# ---------------------------------------------------------------------------
# PPL helpers
# ---------------------------------------------------------------------------

def compute_ppl(model, tokenizer, text: str, context: str = None) -> float:
    """
    Compute perplexity of `text`.
    If `context` is given, tokens are conditioned on context but only
    the loss over `text` tokens is used (teacher-forced, full context).
    """
    if context is not None:
        # Encode context + " " + text, mark boundary
        ctx_ids = tokenizer.encode(context, add_special_tokens=True)
        # Add a space separator so the punchline starts naturally
        full_ids = tokenizer.encode(context + " " + text, add_special_tokens=True)
        n_ctx = len(ctx_ids)
    else:
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        n_ctx = 0

    input_ids = torch.tensor([full_ids])
    n_total = len(full_ids)
    n_target = n_total - n_ctx

    if n_target <= 0:
        return float("nan")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is mean NLL over all tokens; we need only target tokens
        # Recompute per-token logits manually
        logits = outputs.logits  # (1, seq_len, vocab)

    # Shift: logits[i] predicts token[i+1]
    shift_logits = logits[0, :-1, :]   # (seq_len-1, vocab)
    shift_labels = input_ids[0, 1:]    # (seq_len-1,)

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs[range(len(shift_labels)), shift_labels]

    # Only average over punchline tokens (positions n_ctx .. n_total-1).
    # In shifted space, index i predicts token i+1, so token n_ctx is at index
    # n_ctx-1. (With no context, token 0 has no prediction; start at index 0.)
    target_log_probs = token_log_probs[max(n_ctx - 1, 0):]  # punchline tokens

    mean_nll = -target_log_probs.mean().item()
    ppl = np.exp(mean_nll)
    return ppl


def unconditional_ppl(model, tokenizer, punchline: str) -> float:
    return compute_ppl(model, tokenizer, punchline, context=None)


def conditional_ppl(model, tokenizer, setup: str, punchline: str) -> float:
    return compute_ppl(model, tokenizer, punchline, context=setup)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    results = {}  # model_name -> list of ratios (one per joke)

    for model_name in models_to_try:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print('='*60)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            # Sanity check: some checkpoints (notably gpt2-medium) have weights that
            # cause float32 overflow in attention QKV projections (intermediate values
            # reach ~1e35, overflowing when squared). Detect this and retry in float64.
            _test_ids = torch.tensor([[tokenizer.encode("hello")[0]]])
            with torch.no_grad():
                _test_out = model(_test_ids)
            if torch.isnan(_test_out.logits).any():
                print(f"  float32 overflow detected — reloading in float64")
                del model
                model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float64)
                model.eval()
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  dtype: {next(model.parameters()).dtype}")
        except Exception as e:
            print(f"  SKIPPED ({e})")
            continue

        ratios = []
        for joke in jokes:
            ppl_alone = unconditional_ppl(model, tokenizer, joke["punchline"])
            ppl_given_setup = conditional_ppl(model, tokenizer, joke["setup"], joke["punchline"])
            ratio = ppl_given_setup / ppl_alone
            ratios.append(ratio)
            print(f"  [{joke['name']:12s}]  PPL alone={ppl_alone:7.2f}  PPL|setup={ppl_given_setup:7.2f}  ratio={ratio:.4f}")

        results[model_name] = ratios

        # Free memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("SUMMARY TABLE — Misleading Ratio = PPL(punchline|setup) / PPL(punchline alone)")
    print("Ratio > 1.0  →  setup INCREASES surprise (setup successfully misleads)")
    print("Ratio < 1.0  →  setup DECREASES surprise (model treats it as a completion)")
    print("=" * 80)

    col_width = 14
    joke_names = [j["name"] for j in jokes]

    # Header
    header = f"{'Model':<16}" + "".join(f"{name:>{col_width}}" for name in joke_names) + f"{'Avg':>{col_width}}"
    print(header)
    print("-" * len(header))

    for model_name, ratios in results.items():
        avg = np.mean(ratios)
        row = f"{model_name:<16}" + "".join(f"{r:>{col_width}.4f}" for r in ratios) + f"{avg:>{col_width}.4f}"
        print(row)

    # ---------------------------------------------------------------------------
    # Conclusions
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    model_names = list(results.keys())
    avgs = [np.mean(results[m]) for m in model_names]

    if len(avgs) >= 2:
        trend = "INCREASES" if avgs[-1] > avgs[0] else "DECREASES or stays flat"
        print(f"Average misleading ratio trend across scale: {trend}")
        print(f"  {model_names[0]}: {avgs[0]:.4f}  →  {model_names[-1]}: {avgs[-1]:.4f}")
    else:
        print("Only one model loaded; no trend to report.")

    crossed = []
    for model_name, ratios in results.items():
        for i, r in enumerate(ratios):
            if r > 1.0:
                crossed.append((model_name, joke_names[i], r))

    if crossed:
        print("\nModels/jokes where ratio CROSSED 1.0 (setup genuinely misleads):")
        for mn, jn, r in crossed:
            print(f"  {mn} / {jn}: ratio = {r:.4f}")
    else:
        print("\nNo model/joke combination crossed ratio 1.0.")
        print("All models still predict the punchline as a plausible continuation.")


if __name__ == "__main__":
    run_experiment()
