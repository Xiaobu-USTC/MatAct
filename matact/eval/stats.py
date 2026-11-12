"""Statistics used in the paper: paired permutation tests, bootstrap intervals and Holm-Bonferroni correction."""
from __future__ import annotations

import random


def paired_permutation_test(a, b, n_permutations=10000, seed=0) -> float:
    """Two-sided p-value for the mean difference of paired per-task outcomes ``a`` and ``b``."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must have the same non-zero length")
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(sum(diffs))
    hits = 0
    for _ in range(n_permutations):
        if abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_permutations + 1)


def bootstrap_ci(values, n_resamples=10000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval of the mean (task-level resampling)."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return means[int(alpha / 2 * n_resamples)], means[int((1 - alpha / 2) * n_resamples) - 1]


def holm_bonferroni(p_values: dict) -> dict:
    """Holm-adjusted p-values for a family of comparisons {name: p}."""
    order = sorted(p_values, key=p_values.get)
    m, running, adjusted = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        adjusted[name] = running
    return adjusted
