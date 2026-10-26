"""
Experiment M2 Priming: Measuring compression gain from correct joke interpretation.

The hypothesis: under model M2 (the correct interpretation), the full joke becomes
cheaper to encode. We approximate this by prepending a "correct interpretation hint"
and measuring the PPL drop. A big PPL drop = big ΔS (compression gain from M2).
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_model(model_name: str = "distilgpt2"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def ppl_of_text(text: str, model, tokenizer) -> float:
    """Standard whole-text perplexity: exp(mean NLL over all tokens)."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids  # shape: [1, seq_len]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is mean NLL over all tokens (excluding the first,
        # since GPT-2 shifts internally)
        nll = outputs.loss.item()

    return float(np.exp(nll))


def conditional_ppl(prefix: str, target: str, model, tokenizer) -> float:
    """
    Perplexity of `target` tokens only, conditioned on `prefix` context.

    The prefix tokens are fed as context but their NLL is excluded from the average.
    Only the NLL of target tokens contributes to the returned PPL.
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids  # [1, prefix_len]
    target_ids = tokenizer(target, return_tensors="pt").input_ids   # [1, target_len]

    # Concatenate: [prefix | target]
    full_ids = torch.cat([prefix_ids, target_ids], dim=1)  # [1, prefix_len + target_len]

    prefix_len = prefix_ids.shape[1]
    target_len = target_ids.shape[1]

    # Build labels: mask out prefix positions with -100 so they don't contribute to loss
    labels = full_ids.clone()
    labels[0, :prefix_len] = -100  # ignore prefix in loss

    with torch.no_grad():
        outputs = model(full_ids, labels=labels)
        # outputs.loss = mean NLL over non-masked positions
        # GPT2 shifts internally, so position i predicts token i+1.
        # We've masked positions 0..prefix_len-1, so only target tokens are counted.
        nll = outputs.loss.item()

    return float(np.exp(nll))


def main():
    print("Loading distilgpt2 model and tokenizer...")
    model, tokenizer = load_model("distilgpt2")
    print("Model loaded.\n")

    # ------------------------------------------------------------------ #
    # Data: (name, full_joke, m2_hint)
    # ------------------------------------------------------------------ #
    items = [
        (
            "Bungee jump",
            (
                "ME: [licking lips in anticipation] I'm nervous. "
                "I've never done a bungee jump before. "
                "INSTRUCTOR: Don't lick my lips again."
            ),
            "In this story, the speaker has been literally licking the instructor's lips.",
        ),
        (
            "Atoms",
            "Why don't scientists trust atoms? Because they make up everything.",
            "This is a pun joke where 'make up' means both 'invent' and 'constitute'.",
        ),
        (
            "Time flies",
            "Time flies like an arrow. Fruit flies like a banana.",
            "The word 'flies' is a verb in the first sentence but a noun (the insect) in the second.",
        ),
        (
            "Librarian paranoia",
            (
                "I asked the librarian if they had books about paranoia. "
                "She whispered: they're right behind you."
            ),
            "The librarian's whisper refers both to the books' location and to feeding the paranoia.",
        ),
        (
            "Non-joke control",
            "The meeting is scheduled for Tuesday morning. Please confirm your attendance by Friday.",
            "This is a straightforward business communication about scheduling.",
        ),
    ]

    # ------------------------------------------------------------------ #
    # Run experiment
    # ------------------------------------------------------------------ #
    results = []
    for name, joke, hint in items:
        ppl_m1 = ppl_of_text(joke, model, tokenizer)
        ppl_m2 = conditional_ppl(hint, joke, model, tokenizer)
        ratio = ppl_m1 / ppl_m2
        results.append((name, ppl_m1, ppl_m2, ratio))
        print(f"[{name}] PPL(M1)={ppl_m1:.2f}  PPL(M2)={ppl_m2:.2f}  ratio={ratio:.3f}")

    # ------------------------------------------------------------------ #
    # Pretty table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print(f"{'Joke':<22} {'PPL (M1, no hint)':>18} {'PPL (M2, hint)':>15} {'Drop ratio':>12}")
    print("-" * 72)
    for name, ppl_m1, ppl_m2, ratio in results:
        print(f"{name:<22} {ppl_m1:>18.2f} {ppl_m2:>15.2f} {ratio:>12.3f}")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # Discussion
    # ------------------------------------------------------------------ #
    joke_results = results[:-1]
    control = results[-1]

    avg_joke_ratio = np.mean([r[3] for r in joke_results])
    control_ratio = control[3]

    print(
        f"\nAverage drop ratio across jokes : {avg_joke_ratio:.3f}"
    )
    print(f"Drop ratio for non-joke control : {control_ratio:.3f}")
    print()
    print("Discussion")
    print("----------")
    print(
        "The drop ratio (PPL_M1 / PPL_M2) measures how much cheaper a text becomes"
        " to encode once the correct interpretive frame (M2) is supplied. A ratio > 1"
        " means the hint reduces perplexity — the model 'understands' the text better"
        " with the hint."
    )
    print()
    if avg_joke_ratio > control_ratio:
        print(
            f"Result: jokes drop more on average ({avg_joke_ratio:.3f}x) than the"
            f" control ({control_ratio:.3f}x). This is consistent with the M2 hypothesis:"
            " jokes contain deliberate ambiguity that a correct-interpretation hint"
            " resolves more dramatically than a trivially-clear text. The hint shifts"
            " the model from the misleading first-read frame (M1) to the correct one"
            " (M2), yielding a compression gain ΔS ≈ log(ratio)."
        )
    elif avg_joke_ratio < control_ratio:
        print(
            f"Result: the control dropped more ({control_ratio:.3f}x) than jokes on"
            f" average ({avg_joke_ratio:.3f}x). This may indicate that distilgpt2's"
            " perplexity is dominated by surface statistics rather than semantic framing,"
            " limiting its sensitivity to the M1/M2 distinction. A larger, more"
            " context-sensitive model might capture the effect more clearly."
        )
    else:
        print(
            "Result: jokes and control have comparable drop ratios — the perplexity"
            " measure may not be sensitive enough to distinguish joke ambiguity from"
            " ordinary text with this model."
        )
    print()
    print(
        "Note: distilgpt2 is a small, distilled language model. It may not fully"
        " capture the semantic shift from pun/ambiguity resolution. A larger model"
        " (e.g., GPT-2 XL, LLaMA-3) would provide a more reliable signal. However,"
        " even with distilgpt2 the qualitative pattern is informative."
    )


if __name__ == "__main__":
    main()
