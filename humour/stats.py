"""Small-sample group comparisons: Mann-Whitney U, Cliff's delta, bootstrap CI."""

import numpy as np
from scipy.stats import mannwhitneyu


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b) over all pairs. +1 = every a beats every b."""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def bootstrap_ci_delta(a, b, n_boot=5000, seed=7) -> tuple[float, float]:
    """95% percentile-bootstrap CI for Cliff's delta, resampling each group independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    deltas = [cliffs_delta(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    return np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def compare(a, b, ci=False) -> dict:
    """Two-sided Mann-Whitney U plus Cliff's delta (and its bootstrap CI if ci=True)."""
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    out: dict = dict(u=u, p=p, delta=cliffs_delta(a, b))
    if ci:
        out["ci"] = bootstrap_ci_delta(a, b)
    return out


def describe(values) -> str:
    v = np.asarray(values)
    return (f"n={len(v):<3} median={np.median(v):+6.2f}  mean={np.mean(v):+6.2f}  "
            f"IQR=[{np.percentile(v, 25):+.2f}, {np.percentile(v, 75):+.2f}]")
