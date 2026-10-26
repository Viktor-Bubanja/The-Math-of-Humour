"""
Retroactive Compression Experiment
===================================
Tests whether knowing the punchline first makes the setup cheaper to encode
(retroactive compression), as predicted by the information-theoretic view of humor.

Method: Use a left-to-right language model (distilgpt2). Feed [punchline] as context,
then measure PPL of setup tokens conditioned on that context. Compare to unconditional
setup PPL. A ratio > 1 means the setup is cheaper (more compressible) when the
punchline is known — retroactive compression.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "distilgpt2"


def load_model():
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def conditional_ppl(context_text: str, target_text: str, model, tokenizer) -> float:
    """
    Compute PPL of target_text conditioned on context_text.
    Feed context+target as one sequence; compute NLL only over target token positions.
    """
    context_ids = tokenizer.encode(context_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    if len(target_ids) == 0:
        return float("nan")

    full_ids = context_ids + target_ids
    input_ids = torch.tensor([full_ids])

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # The model predicts token[i+1] at position i.
    # Target tokens start at index len(context_ids) in full_ids.
    # The logit for predicting full_ids[j] is at logits[j-1].
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    n_context = len(context_ids)
    n_target = len(target_ids)

    # For each target token at position n_context + k (k=0..n_target-1),
    # the predicting logit is at position (n_context + k - 1).
    nll_total = 0.0
    for k in range(n_target):
        token_id = full_ids[n_context + k]
        pred_pos = n_context + k - 1  # position whose logit predicts this token
        if pred_pos < 0:
            # First target token is the first token overall — use uniform prior
            # (no context at all). This only happens if context is empty.
            nll_total += -np.log(1.0 / tokenizer.vocab_size)
        else:
            nll_total += -log_probs[pred_pos, token_id].item()

    avg_nll = nll_total / n_target
    ppl = np.exp(avg_nll)
    return float(ppl)


def unconditional_ppl(text: str, model, tokenizer) -> float:
    """Standard perplexity with no prefix context."""
    return conditional_ppl("", text, model, tokenizer)


def main():
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

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
            "name": "Librarian",
            "setup": "I asked the librarian if they had books about paranoia. She whispered:",
            "punchline": "they're right behind you.",
        },
        {
            "name": "Control (non-joke)",
            "setup": "The meeting is scheduled for Tuesday morning.",
            "punchline": "Please confirm your attendance by Friday.",
        },
    ]

    results = []
    for joke in jokes:
        name = joke["name"]
        setup = joke["setup"]
        punchline = joke["punchline"]

        ppl_setup_alone = unconditional_ppl(setup, model, tokenizer)
        ppl_setup_given_punchline = conditional_ppl(punchline, setup, model, tokenizer)
        retroactive_ratio = ppl_setup_alone / ppl_setup_given_punchline

        ppl_punchline_alone = unconditional_ppl(punchline, model, tokenizer)
        ppl_punchline_given_setup = conditional_ppl(setup, punchline, model, tokenizer)
        forward_ratio = ppl_punchline_alone / ppl_punchline_given_setup

        results.append(
            {
                "name": name,
                "ppl_setup_alone": ppl_setup_alone,
                "ppl_setup_given_punchline": ppl_setup_given_punchline,
                "retroactive_ratio": retroactive_ratio,
                "ppl_punchline_alone": ppl_punchline_alone,
                "ppl_punchline_given_setup": ppl_punchline_given_setup,
                "forward_ratio": forward_ratio,
            }
        )

    # Print table
    col_widths = [20, 14, 18, 18, 16, 18, 14]
    headers = [
        "Joke",
        "PPL(setup)",
        "PPL(setup|punch)",
        "Retro ratio",
        "PPL(punch)",
        "PPL(punch|setup)",
        "Fwd ratio",
    ]

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_row = "|" + "|".join(
        h.center(w) for h, w in zip(headers, col_widths)
    ) + "|"

    print(sep)
    print(header_row)
    print(sep)

    for r in results:
        row = "|" + "|".join([
            r["name"].center(col_widths[0]),
            f"{r['ppl_setup_alone']:.2f}".center(col_widths[1]),
            f"{r['ppl_setup_given_punchline']:.2f}".center(col_widths[2]),
            f"{r['retroactive_ratio']:.3f}".center(col_widths[3]),
            f"{r['ppl_punchline_alone']:.2f}".center(col_widths[4]),
            f"{r['ppl_punchline_given_setup']:.2f}".center(col_widths[5]),
            f"{r['forward_ratio']:.3f}".center(col_widths[6]),
        ]) + "|"
        print(row)

    print(sep)

    # Analysis
    print("\n=== Analysis ===\n")

    jokes_results = [r for r in results if "Control" not in r["name"]]
    control_results = [r for r in results if "Control" in r["name"]]

    avg_retro_jokes = np.mean([r["retroactive_ratio"] for r in jokes_results])
    avg_fwd_jokes = np.mean([r["forward_ratio"] for r in jokes_results])
    avg_retro_control = np.mean([r["retroactive_ratio"] for r in control_results])
    avg_fwd_control = np.mean([r["forward_ratio"] for r in control_results])

    print(f"Jokes (n={len(jokes_results)}):")
    print(f"  Avg retroactive ratio (PPL_setup / PPL_setup|punch): {avg_retro_jokes:.3f}")
    print(f"  Avg forward ratio     (PPL_punch / PPL_punch|setup): {avg_fwd_jokes:.3f}")
    print(f"\nControl (n={len(control_results)}):")
    print(f"  Avg retroactive ratio: {avg_retro_control:.3f}")
    print(f"  Avg forward ratio:     {avg_fwd_control:.3f}")

    print("\n=== Conclusions ===\n")

    retro_gt_1_count = sum(1 for r in jokes_results if r["retroactive_ratio"] > 1)
    print(
        f"Retroactive ratio > 1 (setup cheaper when punchline known) in "
        f"{retro_gt_1_count}/{len(jokes_results)} jokes."
    )

    if avg_retro_jokes > avg_retro_control:
        print(
            f"Jokes show MORE retroactive compression than control "
            f"({avg_retro_jokes:.3f} vs {avg_retro_control:.3f}). "
            "This supports the blog's prediction."
        )
    else:
        print(
            f"Jokes show LESS retroactive compression than control "
            f"({avg_retro_jokes:.3f} vs {avg_retro_control:.3f}). "
            "The blog's prediction is NOT supported for this model/joke set."
        )

    if avg_retro_jokes > avg_fwd_jokes:
        print(
            f"For jokes, retroactive ratio ({avg_retro_jokes:.3f}) > forward ratio ({avg_fwd_jokes:.3f}): "
            "knowing the punchline helps the setup MORE than knowing the setup helps the punchline."
        )
    elif avg_retro_jokes < avg_fwd_jokes:
        print(
            f"For jokes, retroactive ratio ({avg_retro_jokes:.3f}) < forward ratio ({avg_fwd_jokes:.3f}): "
            "knowing the setup helps the punchline more (standard forward coherence dominates)."
        )
    else:
        print("Retroactive and forward ratios are approximately equal.")

    print("\nNote: Ratio > 1 means the context lowers PPL (compresses the text).")
    print("Ratio < 1 means the context raises PPL (the context misleads the model).")


if __name__ == "__main__":
    main()
