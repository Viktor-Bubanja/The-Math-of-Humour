import math

import pytest
import torch

from humour.corpus import ABSURDIST, FAMOUS_JOKES, FOILS, ITEMS, NOVEL_JOKES
from humour.stats import cliffs_delta, paired_compare, rank_biserial
from humour.surprisal import bos_ids, continuation_surprisals, load, punchline_surprisal

# Every prefix any experiment uses ("" = no frame).
FRAMES = [
    "", "Here is a well-known joke: ", "Here is a routine factual note: ",
    "The following text is a joke.\n", "The following text is a factual statement.\n",
    "Joke:\n", "Note:\n", "What follows is meant to be funny. ", "What follows is meant to inform. ",
    "You are about to read a joke. ", "You are about to read a factual memo. ",
    "The following text is absurdist anti-humour, a deliberate performance of nonsense.\n",
]
TEXTS = ([(s, p) for _, _, s, p in ITEMS + NOVEL_JOKES] + [(s, p) for _, s, p in ABSURDIST]
         + [(s, foil) for n, _, s, _ in FAMOUS_JOKES + NOVEL_JOKES for foil in FOILS[n]])


@pytest.fixture(scope="module")
def distilgpt2():
    return load("distilgpt2")


@pytest.mark.parametrize("model_name", ["distilgpt2", "gpt2-large", "google/gemma-2-2b-it"])
def test_punchline_boundary_tokenizes_like_running_text(model_name):
    """punchline_surprisal tokenizes setup and punchline separately. That is only valid
    if the split gives the same tokens as tokenizing the whole text: a BPE merge across
    the boundary would silently change which tokens count as the punchline."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except OSError:
        pytest.skip(f"{model_name} tokenizer not available (gated or not downloaded)")
    for frame in FRAMES:
        for setup, punch in TEXTS:
            split = (tok(frame + setup, add_special_tokens=False)["input_ids"]
                     + tok(punch, add_special_tokens=False)["input_ids"])
            joint = tok(frame + setup + punch, add_special_tokens=False)["input_ids"]
            assert split == joint, (frame, setup, punch)


def test_matches_huggingface_loss(distilgpt2):
    """Mean surprisal in bits, converted to nats, equals HF's own cross-entropy over the
    same target tokens (context positions masked out with -100)."""
    model, tok = distilgpt2
    context = bos_ids(tok) + tok("Time flies like an arrow.", add_special_tokens=False)["input_ids"]
    target = tok(" Fruit flies like a banana.", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([context + target])
    labels = ids.clone()
    labels[0, :len(context)] = -100
    with torch.no_grad():
        hf_nats = model(ids, labels=labels).loss.item()
    bits = continuation_surprisals(model, context, target)
    assert sum(bits) / len(bits) * math.log(2) == pytest.approx(hf_nats, rel=1e-5)


def test_frame_changes_only_the_context(distilgpt2):
    model, tok = distilgpt2
    setup, punch = "I used to be a banker.", " But I lost interest."
    plain = punchline_surprisal(model, tok, "", setup, punch)
    framed = punchline_surprisal(model, tok, "Joke:\n", setup, punch)
    assert plain > 0 and framed > 0
    assert plain != framed


def test_cliffs_delta():
    assert cliffs_delta([3, 4], [1, 2]) == 1
    assert cliffs_delta([1, 2], [3, 4]) == -1
    assert cliffs_delta([1, 2], [1, 2]) == 0


def test_paired_stats():
    assert rank_biserial([1, 2, 3]) == 1
    assert rank_biserial([-1, -2, 3]) == pytest.approx((3 - 3) / 6)
    c = paired_compare([5, 6, 7, 8, 9, 10], [1, 1, 1, 1, 1, 1])
    assert c["median"] == 6.5 and c["r"] == 1 and c["ci"][0] > 0

