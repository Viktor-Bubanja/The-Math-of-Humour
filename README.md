# The Math of Humour

Code and data for the blog post *Laughter as a Compression Dividend*. The post treats a joke as model selection in Minimum Description Length terms: the setup makes you adopt one interpretation, the punchline is expensive under it, and the "click" is switching to a second interpretation that makes the whole text cheap to encode. This repo uses small language models as a measuring instrument to see whether those predictions show up in per-token surprisal.

The short version of what came out:

- **Telling a model "this is a joke" makes joke punchlines cheaper.** Prefix every text with "this is a joke" vs "this is a factual note" and joke punchlines get cheaper while prose gets more expensive (gemma-2-2b-it: p = 0.001, Cliff's δ = +0.72). The effect survives newly written jokes the model can't have memorised, five different frame wordings, and two models.
- **In gemma-2-2b-it the frame favours punchlines that resolve the joke.** Swap a punchline for a matched foil that is at least as surprising but makes no sense of the setup, and the frame helps the foil less (E19). distilgpt2 shows the opposite, which suggests its frame effect is closer to "expect something odd".
- **The surprisal spike itself doesn't mark a joke.** Peak height, spike sharpness, recovery speed and "context damage" don't separate jokes from prose at corpus scale (37 texts, three models); an early five-text result that suggested otherwise was a small-sample artifact.

Everything is pilot-scale (dozens of texts, models up to 2.6B parameters). Read the results as a careful hobby experiment, not a benchmark.

## Results

| # | Question | Model(s) | Result |
|---|---|---|---|
| E11 | Do peak surprisal / sharpness / recovery separate jokes from prose? | distilgpt2 | No: peak p ≥ 0.14, sharpness p ≥ 0.20; 32/37 texts "recover" in one token |
| E12 | Does the setup make the twist token *harder* to predict than no context? | distilgpt2, gpt2-large | No: weak trend at 82M (p = 0.06), gone at 774M |
| E13 | Same, at a larger instruction-tuned model | gemma-2-2b-it | No: p ≥ 0.49 |
| E14 | Does a "joke" frame make joke punchlines cheaper than a "factual" frame, relative to prose? | distilgpt2, gemma-2-2b-it | **Yes**: p = 0.001 in both, effect ~4× larger in gemma |
| E15 | Same with a blander wording | distilgpt2, gemma-2-2b-it | Gemma loses significance on these famous jokes (p = 0.20), which suggests a memorisation confound. distilgpt2 still separates (p = 0.016) |
| E16 | Does the effect survive on 12 newly written jokes? | distilgpt2, gemma-2-2b-it | **Yes**: all 4 model × wording conditions, p = 0.004–0.038, δ = +0.53 to +0.73, bootstrap CIs exclude 0 |
| E17 | Does a "deliberate nonsense" frame specifically compress absurdist texts? | gemma-2-2b-it | Directionally yes (p = 0.008, n = 10); pilot only |
| E18 | Is E16 an artifact of one prompt wording? | gemma-2-2b-it | **No**: holds in 5/5 phrasings, p = 0.0009–0.027 |
| E19 | Does the frame favour a joke's real punchline over a matched foil that is at least as surprising but doesn't resolve the joke? | distilgpt2, gemma-2-2b-it | **Gemma: yes**. Real − foil benefit median +0.21 bits/token [95% CI +0.08, +0.38], p = 0.0001, 28/39 jokes; novel jokes alone p = 0.016. Holds in 4/5 phrasings (not "funny"). **distilgpt2: reversed** (−0.07, p = 0.03): its frame favours the stranger foil |

Per-item numbers are in [`results/`](results/) alongside every figure.

## Method

Surprisal of token *t* is −log₂ p(*t* | everything before it), i.e. the number of bits an ideal code built from the model spends on *t*. Every experiment is a variation on one function ([`humour/surprisal.py`](humour/surprisal.py)):

```python
benefit = (punchline_surprisal(model, tok, FRAME_MEMO, setup, punch)
         - punchline_surprisal(model, tok, FRAME_JOKE, setup, punch))
```

`punchline_surprisal` runs `BOS + prefix + setup + punchline` through the model in one teacher-forced pass and averages the punchline tokens' surprisal. The frame experiments then test whether `benefit` is larger for jokes than for prose (two-sided Mann-Whitney U, Cliff's δ, percentile-bootstrap CIs; [`humour/stats.py`](humour/stats.py)). Because the test is a jokes-vs-prose interaction, any effect of the prefixes that isn't joke-specific cancels out. E19 instead compares each joke's real punchline with its own matched foils (two-sided Wilcoxon signed-rank, matched-pairs rank-biserial r, bootstrap CI over jokes).

### Validity checks

| Threat | Check |
|---|---|
| Categories chosen to fit results | Categories and punchline boundaries fixed before any model run ([`humour/corpus.py`](humour/corpus.py)) |
| Per-item wording leaking hints | One uniform prefix for every text |
| Memorised famous jokes | 12 new jokes (E16) |
| Prompt sensitivity | 5 frame phrasings (E18) |
| Single-model quirks | 3 models, 82M to 2.6B parameters |
| Tokenizer merges across the setup/punchline boundary | [`tests/`](tests/) checks every frame × text pair, including E19's foils, under all three tokenizers |
| "The frame just likes surprising text" | E19: each joke's real punchline vs a matched foil of the same length and shape that is at least as surprising with no frame, but doesn't resolve the joke. Foils fixed and reviewed before any framed run |
| Off-by-one in reading probabilities | [`tests/`](tests/) checks surprisal against Hugging Face's own cross-entropy loss |
| Non-reproducible numbers | Pinned dependencies. Re-running `experiments/` reproduces the original CSVs byte-for-byte (E15 now also adds distilgpt2 rows) |

Two deliberate choices worth knowing about: `gemma-2-2b-it` gets raw text, not its chat template (the template would itself act as a frame), and all runs are on CPU (Gemma in bfloat16), so results don't depend on GPU kernels.

## Running it

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests                     # ~15s
.venv/bin/python experiments/e14_genre_frame.py      # writes results/genre_frame_results.csv + figure
```

`google/gemma-2-2b-it` is gated on Hugging Face: accept its licence and `huggingface-cli login` first. The distilgpt2 scripts take seconds. The Gemma ones take roughly 10–20 minutes each on a laptop CPU; E19, which scores 117 punchlines under 11 conditions, takes closer to an hour.

| Script | Experiments |
|---|---|
| `experiments/e11_corpus_sharpness.py` | E11 |
| `experiments/e12_context_effect.py` | E12; pass `google/gemma-2-2b-it` for E13 |
| `experiments/e14_genre_frame.py` | E14; pass `neutral` for E15 |
| `experiments/e16_novel_jokes.py` | E16 |
| `experiments/e17_absurdism_frame.py` | E17 |
| `experiments/e18_frame_sweep.py` | E18 |
| `experiments/e19_minimal_pairs.py` | E19 |

## Layout

```
humour/          shared code: corpus, surprisal, stats, plotting
experiments/     E11–E19, the experiments the post reports
exploratory/     E1–E10, early single-file probes on 4–5 texts each (superseded; kept for the record)
results/         CSVs and figures written by experiments/
tests/           tokenization-boundary, loss-equivalence and statistics checks
```

### Exploratory scripts (E1–E10)

These came first and ran on a handful of hand-picked texts. Several looked like strong results, and most didn't survive E11's move to a 37-text corpus with controls. They're kept as-is apart from one bug fix: `e08_model_scale.py` read punchline probabilities one position late and skipped the first punchline token. None of its numbers appear in the post.

| # | Probe | Outcome at the time |
|---|---|---|
| E1 | Setup-misleading ratio, PPL(punch \| setup) / PPL(punch) | Null |
| E2 | Token-level surprisal profile | Spike on the twist word (later shown to be generic, E11) |
| E3 | Embedding distance between punchline and sampled continuations | Null (shared prefix dominates the embedding) |
| E4 | Gini coefficient of whole-text surprisal | Null |
| E5 | Perplexity drop from an interpretation hint | Directional only |
| E6 | Post-peak recovery / "resolution sharpness" | Looked strong at n = 5, null at n = 37 (E11) |
| E7 | Retroactive compression of the setup given the punchline | Not joke-specific |
| E8 | Misleading ratio across GPT-2 sizes | Rises with scale, never crosses 1 |
| E9 | Gini of punchline tokens only | Null |
| E10 | Actual punchline vs greedy completion | Bungee punchline ~39× less probable per token (anecdote in the post) |
