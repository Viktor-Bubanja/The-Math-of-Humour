"""
Experiment: Resolution Sharpness in Humor
==========================================
Quantifies how cleanly and quickly surprisal collapses after the peak twist token
in jokes, testing the hypothesis that better jokes have sharper, faster recovery.
"""

import torch
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

MODEL_NAME = "distilgpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def token_surprisals(full_text: str) -> list[tuple[str, float]]:
    """Return list of (token_string, surprisal_bits) for every token in full_text."""
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]  # shape: (1, T)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape: (1, T, V)

    # Surprisal of token t is -log2 P(token_t | token_0..t-1)
    # logits[:, t, :] gives the distribution for position t+1
    # So to get surprisal of token at index t (t >= 1), use logits[:, t-1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (1, T, V)

    tokens = input_ids[0]  # (T,)
    result = []

    for t in range(len(tokens)):
        token_str = tokenizer.decode([tokens[t].item()])
        if t == 0:
            # First token has no context — we can estimate its unigram surprisal
            # using the model's distribution given an empty-ish context.
            # Conventional approach: treat position 0 separately; assign NaN or 0.
            # We'll use the model's prior (logits at position 0 from a BOS-primed run).
            # For consistency and simplicity, skip t=0 surprisal computation by
            # using the logits from the model's unconditional output.
            lp = log_probs[0, 0, tokens[0].item()].item()  # P(token_0 | nothing useful)
            # This is effectively unconditional from model perspective at t=0
            surprisal = -lp / np.log(2)
        else:
            lp = log_probs[0, t - 1, tokens[t].item()].item()
            surprisal = -lp / np.log(2)

        result.append((token_str, surprisal))

    return result


def analyze_joke(name: str, full: str, setup: str, punchline: str) -> dict:
    """
    Analyze resolution sharpness metrics for a joke.
    Returns a dict with all metrics and the full token sequence.
    """
    # Tokenize consistently
    setup_ids = tokenizer(setup, add_special_tokens=False)["input_ids"]
    full_ids  = tokenizer(full,  add_special_tokens=False)["input_ids"]

    n_setup = len(setup_ids)
    # Verify: the first n_setup tokens of full should match setup_ids
    assert full_ids[:n_setup] == setup_ids, (
        f"[{name}] Setup tokenization mismatch!\n"
        f"  setup_ids[:5]={setup_ids[:5]}\n"
        f"  full_ids[:5]={full_ids[:5]}"
    )

    # Get per-token surprisals for full text
    surprisals = token_surprisals(full)  # list of (str, float), length == len(full_ids)

    setup_surprisals     = surprisals[:n_setup]
    punchline_surprisals = surprisals[n_setup:]

    # Pre-peak mean: mean over all setup tokens
    pre_peak_mean = float(np.mean([s for _, s in setup_surprisals]))

    # Peak in punchline
    if not punchline_surprisals:
        raise ValueError(f"[{name}] No punchline tokens found.")

    punch_vals  = [s for _, s in punchline_surprisals]
    peak_idx    = int(np.argmax(punch_vals))           # index within punchline
    peak_surprisal = punch_vals[peak_idx]
    peak_token  = punchline_surprisals[peak_idx][0]

    # Post-peak: tokens in punchline AFTER the peak
    post_peak_vals = punch_vals[peak_idx + 1:]

    if post_peak_vals:
        post_peak_mean = float(np.mean(post_peak_vals))
        resolution_sharpness = peak_surprisal / post_peak_mean if post_peak_mean > 0 else float("inf")
    else:
        post_peak_mean = float("nan")
        resolution_sharpness = float("nan")

    # Recovery to baseline
    recovery_to_baseline = (not np.isnan(post_peak_mean)) and (post_peak_mean < pre_peak_mean)

    # Tokens to recover: how many tokens after peak until surprisal first drops below pre_peak_mean
    tokens_to_recover = None
    for i, v in enumerate(post_peak_vals):
        if v < pre_peak_mean:
            tokens_to_recover = i + 1  # 1-indexed count of tokens elapsed
            break

    return {
        "name": name,
        "full_text": full,
        "surprisals": surprisals,
        "n_setup": n_setup,
        "pre_peak_mean": pre_peak_mean,
        "peak_surprisal": peak_surprisal,
        "peak_token": peak_token,
        "peak_idx_in_punchline": peak_idx,
        "post_peak_mean": post_peak_mean,
        "resolution_sharpness": resolution_sharpness,
        "recovery_to_baseline": recovery_to_baseline,
        "tokens_to_recover": tokens_to_recover,
    }


# ---------------------------------------------------------------------------
# Jokes
# ---------------------------------------------------------------------------

jokes = [
    dict(
        name="Time flies",
        full="Time flies like an arrow. Fruit flies like a banana.",
        setup="Time flies like an arrow.",
        punchline=" Fruit flies like a banana.",
    ),
    dict(
        name="Bungee",
        full="ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR: Don't lick my lips again.",
        setup="ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR:",
        punchline=" Don't lick my lips again.",
    ),
    dict(
        name="Atoms",
        full="Why don't scientists trust atoms? Because they make up everything.",
        setup="Why don't scientists trust atoms?",
        punchline=" Because they make up everything.",
    ),
    dict(
        name="Librarian",
        full="I asked the librarian if they had books about paranoia. She whispered: they're right behind you.",
        setup="I asked the librarian if they had books about paranoia. She whispered:",
        punchline=" they're right behind you.",
    ),
    dict(
        name="Control (non-joke)",
        full="The meeting is scheduled for Tuesday morning. Please confirm your attendance by Friday.",
        setup="The meeting is scheduled for Tuesday morning.",
        punchline=" Please confirm your attendance by Friday.",
    ),
]

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

results = []
for j in jokes:
    print(f"  Analyzing: {j['name']} ...", flush=True)
    r = analyze_joke(j["name"], j["full"], j["setup"], j["punchline"])
    results.append(r)

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

print()
print("=" * 110)
print("RESULTS TABLE")
print("=" * 110)

header = (
    f"{'Joke':<22} {'Peak Surp':>10} {'Peak Token':<18} {'Pre-Peak':>9} "
    f"{'Post-Peak':>10} {'Sharpness':>10} {'Recovered?':>11} {'Tok2Recov':>10}"
)
print(header)
print("-" * 110)

for r in results:
    tok2r = str(r["tokens_to_recover"]) if r["tokens_to_recover"] is not None else "never"
    pp_mean = f"{r['post_peak_mean']:.2f}" if not np.isnan(r["post_peak_mean"]) else "n/a"
    sharp   = f"{r['resolution_sharpness']:.2f}" if not np.isnan(r["resolution_sharpness"]) else "n/a"
    row = (
        f"{r['name']:<22} {r['peak_surprisal']:>10.2f} {repr(r['peak_token']):<18} "
        f"{r['pre_peak_mean']:>9.2f} {pp_mean:>10} {sharp:>10} "
        f"{'yes' if r['recovery_to_baseline'] else 'no':>11} {tok2r:>10}"
    )
    print(row)

# ---------------------------------------------------------------------------
# Print token-by-token sequences
# ---------------------------------------------------------------------------

print()
print("=" * 110)
print("TOKEN-BY-TOKEN SURPRISAL (setup || punchline)")
print("=" * 110)

for r in results:
    print(f"\n--- {r['name']} ---")
    print(f"Text: {r['full_text'][:100]}{'...' if len(r['full_text'])>100 else ''}")
    print()

    tokens_list = r["surprisals"]
    n_setup     = r["n_setup"]

    # Build column-aligned output
    col_w = 18
    line1 = ""  # token
    line2 = ""  # surprisal

    for i, (tok, surp) in enumerate(tokens_list):
        if i == n_setup:
            sep = " || "
            line1 += sep
            line2 += sep

        tok_disp  = repr(tok)[:col_w - 1].ljust(col_w)
        surp_disp = f"{surp:.2f}".rjust(col_w)
        line1 += tok_disp
        line2 += surp_disp

    print("Tokens:     " + line1)
    print("Surp (bits):" + line2)
    print()

    # Also print peak position
    punch_vals = [s for _, s in tokens_list[n_setup:]]
    if punch_vals:
        peak_offset = int(np.argmax(punch_vals))
        print(f"  Peak in punchline at offset {peak_offset}: token={repr(tokens_list[n_setup + peak_offset][0])!r}, "
              f"surprisal={tokens_list[n_setup + peak_offset][1]:.2f} bits")
    print()

# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------

print("=" * 110)
print("CONCLUSION")
print("=" * 110)

# Sort by resolution_sharpness (ignoring nan)
ranked = sorted(
    [r for r in results if not np.isnan(r["resolution_sharpness"])],
    key=lambda r: r["resolution_sharpness"],
    reverse=True
)

print()
print("Jokes ranked by resolution sharpness (peak_surprisal / post_peak_mean):")
for i, r in enumerate(ranked, 1):
    tok2r = f"{r['tokens_to_recover']} tokens" if r["tokens_to_recover"] is not None else "never recovers"
    print(f"  {i}. {r['name']:<22}  sharpness={r['resolution_sharpness']:.2f}  "
          f"recovered={'yes' if r['recovery_to_baseline'] else 'no'}  ({tok2r})")

print()
print("Interpretation:")
sharpest = ranked[0]
print(f"  - Sharpest resolution: '{sharpest['name']}' (sharpness={sharpest['resolution_sharpness']:.2f})")
print(f"    This means the peak surprisal is {sharpest['resolution_sharpness']:.1f}x the mean post-peak surprisal,")
print(f"    indicating the model snaps very cleanly into the alternative parse after the twist token.")
print()

# Check if better jokes (non-control) tend to recover faster
joke_results = [r for r in results if r["name"] != "Control (non-joke)"]
control_r    = next(r for r in results if r["name"] == "Control (non-joke)")

joke_sharp_mean    = np.mean([r["resolution_sharpness"] for r in joke_results
                               if not np.isnan(r["resolution_sharpness"])])
control_sharp      = control_r["resolution_sharpness"]

print(f"  - Mean sharpness across jokes (excl. control): {joke_sharp_mean:.2f}")
print(f"  - Control (non-joke) sharpness:                 {control_sharp:.2f}")
if joke_sharp_mean > control_sharp:
    print("  - HYPOTHESIS SUPPORTED: Jokes show sharper resolution than the control text,")
    print("    consistent with the idea that humor requires — and achieves — a clean M1→M2 parse switch.")
else:
    print("  - HYPOTHESIS NOT CLEARLY SUPPORTED: Jokes do not uniformly show sharper resolution than control.")
    print("    This may reflect distilGPT2's limited ability to represent human joke-parsing expectations.")
