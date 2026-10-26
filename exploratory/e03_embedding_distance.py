"""
Experiment: Measuring joke punchline off-manifold distance in embedding space.

The hypothesis: jokes work geometrically — the setup primes a semantic manifold M₁,
and the punchline lands far off it. We test this by generating K completions of the
setup with distilgpt2, then measuring how far the actual punchline sits from the
centroid of those completions in embedding space.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)

MODEL_NAME = "distilgpt2"
K = 10
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 15

jokes = [
    {
        "name": "Bungee jump",
        "setup": "ME: [licking lips in anticipation] I'm nervous. I've never done a bungee jump before. INSTRUCTOR:",
        "punchline": "Don't lick my lips again.",
    },
    {
        "name": "Atoms",
        "setup": "Why don't scientists trust atoms?",
        "punchline": "Because they make up everything.",
    },
    {
        "name": "Time flies",
        "setup": "Time flies like an arrow.",
        "punchline": "Fruit flies like a banana.",
    },
    {
        "name": "Non-joke control",
        "setup": "The meeting is scheduled for Tuesday morning.",
        "punchline": "Please confirm your attendance by Friday.",
    },
]


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def get_embedding(text: str, tokenizer, model) -> np.ndarray:
    """Embed text using mean of last-layer hidden states."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.hidden_states is a tuple of (num_layers+1) tensors, each (batch, seq_len, hidden_dim)
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    embedding = last_hidden.mean(dim=1).squeeze(0).numpy()  # (hidden_dim,)
    return embedding


def generate_completions(setup: str, tokenizer, model, k: int) -> list[str]:
    """Generate K completions of the setup using sampling."""
    inputs = tokenizer(setup, return_tensors="pt")
    input_ids = inputs["input_ids"]
    completions = []
    with torch.no_grad():
        for _ in range(k):
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            new_tokens = output[0][input_ids.shape[1]:]
            completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(setup + completion_text)
    return completions


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


def avg_pairwise_distance(embeddings: list[np.ndarray]) -> float:
    """Average cosine distance between all pairs of embeddings."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine_distance(embeddings[i], embeddings[j])
            count += 1
    return total / count


def run_experiment(joke: dict, tokenizer, model) -> dict:
    name = joke["name"]
    setup = joke["setup"]
    punchline = joke["punchline"]

    print(f"  Generating {K} completions for: {name}...")
    completions = generate_completions(setup, tokenizer, model, K)

    print(f"  Embedding completions and punchline...")
    completion_embeddings = [get_embedding(c, tokenizer, model) for c in completions]
    punchline_embedding = get_embedding(setup + " " + punchline, tokenizer, model)

    centroid = np.mean(completion_embeddings, axis=0)

    punchline_dist = cosine_distance(punchline_embedding, centroid)
    inter_dist = avg_pairwise_distance(completion_embeddings)
    ratio = punchline_dist / inter_dist if inter_dist > 0 else float("inf")

    return {
        "name": name,
        "punchline_distance": punchline_dist,
        "avg_inter_completion_distance": inter_dist,
        "ratio": ratio,
        "completions_sample": completions[:2],  # show first 2 for context
    }


def print_results(results: list[dict]):
    print("\n" + "=" * 80)
    print("RESULTS: Joke Punchline Off-Manifold Distance")
    print("=" * 80)
    print(f"{'Joke':<22} {'Punchline→Centroid':>20} {'Avg Inter-Completion':>22} {'Ratio':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<22} {r['punchline_distance']:>20.4f} {r['avg_inter_completion_distance']:>22.4f} {r['ratio']:>8.2f}x"
        )
    print("=" * 80)
    print()
    print("Interpretation:")
    print("  Ratio >> 1  → punchline lands far off the model's expected manifold (joke-like)")
    print("  Ratio ~  1  → punchline is about as unexpected as any other completion")
    print("  Ratio <  1  → punchline sits close to the expected manifold (control-like)")
    print()
    print("Sample completions (first 2) per joke:")
    print("-" * 80)
    for r in results:
        print(f"\n[{r['name']}]")
        for i, c in enumerate(r["completions_sample"], 1):
            # Truncate long lines for display
            display = c if len(c) <= 120 else c[:117] + "..."
            print(f"  Completion {i}: {display}")


def main():
    tokenizer, model = load_model()

    all_results = []
    for joke in jokes:
        print(f"\nProcessing: {joke['name']}")
        result = run_experiment(joke, tokenizer, model)
        all_results.append(result)

    print_results(all_results)


if __name__ == "__main__":
    main()
