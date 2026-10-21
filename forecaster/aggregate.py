"""Combining the ensemble's forecasts into one."""

from __future__ import annotations

import statistics

import numpy as np

from . import cdf as cdfs
from .metaculus import Question

BINARY_MIN = 0.02
BINARY_MAX = 0.98
CHOICE_FLOOR = 0.01


def binary(probabilities: list[float]) -> float:
    if not probabilities:
        raise ValueError("no forecasts")
    return float(min(max(statistics.median(probabilities), BINARY_MIN), BINARY_MAX))


def multiple_choice(distributions: list[list[float]]) -> list[float]:
    if not distributions:
        raise ValueError("no forecasts")
    columns = list(zip(*distributions))
    medians = [statistics.median(c) for c in columns]
    total = sum(medians) or 1.0
    probs = [m / total for m in medians]
    for _ in range(10):
        probs = [max(p, CHOICE_FLOOR) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]
        if min(probs) >= CHOICE_FLOOR - 1e-12:
            break
    return probs


def continuous(member_cdfs: list[np.ndarray], q: Question) -> list[float]:
    """The mean of the members' CDFs, lightly smoothed, then made valid for Metaculus."""
    if not member_cdfs:
        raise ValueError("no forecasts")
    mean = np.mean(np.vstack(member_cdfs), axis=0)
    if q.type != "discrete":
        mean = cdfs.smooth(mean)
    result = cdfs.standardize(mean, q)
    issues = cdfs.problems(result, q)
    if issues:
        raise ValueError("invalid cdf")
    return result
