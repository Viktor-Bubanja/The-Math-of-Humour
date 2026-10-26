"""Per-token surprisal (bits) from a causal language model.

Surprisal of token t is -log2 p(t | everything before it): the number of bits
an ideal code built from the model would spend on t. All experiments reduce
to three operations on top of that:

  - token_surprisals:        every token of a text (E11's cliff plots)
  - continuation_surprisals: target tokens given an arbitrary context (E12)
  - punchline_surprisal:     mean over a punchline given prefix + setup (E14+)
"""

import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_name: str):
    """Load a model + tokenizer in eval mode. Gemma runs in bfloat16, GPT-2 in float32."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if "gemma" in model_name else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.eval()
    return model, tokenizer


def bos_ids(tokenizer) -> list[int]:
    """[BOS] if the tokenizer has one (<|endoftext|> for GPT-2, <bos> for Gemma), else []."""
    return [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []


def _bits(model, input_ids: list[int], positions: range) -> list[float]:
    """Surprisal in bits of input_ids[i] for each i in positions (all i >= 1)."""
    with torch.no_grad():
        logits = model(torch.tensor([input_ids])).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0]
    # logits at position i predict token i + 1
    pred = torch.tensor([i - 1 for i in positions])
    target = torch.tensor([input_ids[i] for i in positions])
    # .double() before dividing so bf16 log-probs are converted exactly, not re-rounded
    return (-log_probs[pred, target].double() / math.log(2)).tolist()


def token_surprisals(model, tokenizer, text: str) -> list[tuple[str, float]]:
    """(token, bits) for every token of text, with no BOS. Token 0 has no context -> NaN."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bits = [float("nan")] + _bits(model, ids, range(1, len(ids)))
    return [(tokenizer.decode([t]), b) for t, b in zip(ids, bits)]


def continuation_surprisals(model, context_ids: list[int], target_ids: list[int]) -> list[float]:
    """Bits for each target token, teacher-forced after context_ids (which must be non-empty)."""
    n_ctx = len(context_ids)
    return _bits(model, context_ids + target_ids, range(n_ctx, n_ctx + len(target_ids)))


def punchline_surprisal(model, tokenizer, prefix: str, setup: str, punch: str) -> float:
    """Mean bits/token of punch, read after BOS + prefix + setup.

    prefix + setup is tokenized as one string so a frame like "Joke:\\n" joins the
    setup exactly as it would in running text. punch starts with a space, so
    tokenizing it separately matches how it tokenizes in context.
    """
    context = bos_ids(tokenizer) + tokenizer(prefix + setup, add_special_tokens=False)["input_ids"]
    punch_ids = tokenizer(punch, add_special_tokens=False)["input_ids"]
    return float(np.mean(continuation_surprisals(model, context, punch_ids)))
