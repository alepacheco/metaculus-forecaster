"""Continuous forecasts: percentiles to a Metaculus CDF, and the CDF rules.

A numeric, discrete or date question takes a CDF of `inbound_outcome_count + 1` points spread
evenly over the question's scaled range (linear, or logarithmic when the question has a zero
point). The rules (from the Metaculus API documentation): no mass outside a closed bound, at
least 0.1% outside an open one, at least a 1% uniform floor spread over the range, and no single
bin above 0.2 x 200 / inbound_outcome_count. The standardisation below follows the same steps
as the MIT-licensed forecasting-tools library (see NOTICE).
"""

from __future__ import annotations

import math

import numpy as np

from .metaculus import Question

MAX_BIN_AT_200 = 0.2
BIN_CAP_MARGIN = 0.95


def value_to_location(value: float, q: Question) -> float:
    lo, hi, zero = q.range_min, q.range_max, q.zero_point
    if zero is None:
        return (value - lo) / (hi - lo)
    ratio = (hi - zero) / (lo - zero)
    if value == zero:
        value += 1e-10
    inner = (value - lo) * (ratio - 1) + (hi - lo)
    if inner <= 0:
        return -1.0
    return (math.log(inner) - math.log(hi - lo)) / math.log(ratio)


def location_to_value(location: float, q: Question) -> float:
    lo, hi, zero = q.range_min, q.range_max, q.zero_point
    if zero is None:
        return lo + (hi - lo) * location
    ratio = (hi - zero) / (lo - zero)
    return lo + (hi - lo) * (ratio ** location - 1) / (ratio - 1)


def bin_cap(q: Question) -> float:
    return MAX_BIN_AT_200 * 200 / (q.cdf_size - 1)


def percentiles_to_cdf(points: list[tuple[float, float]], q: Question) -> np.ndarray:
    """A raw CDF on the question's grid from (percentile in 0..1, value) points.

    Between declared points the CDF is linear in scaled location; beyond them it continues with
    the slope of the outermost segment until it reaches 0 or 1.
    """
    pts = sorted(points, key=lambda p: p[1])
    locs = np.array([value_to_location(v, q) for _, v in pts], dtype=float)
    ps = np.array([p for p, _ in pts], dtype=float)
    if len(pts) < 2 or np.any(np.diff(locs) <= 0) or np.any(np.diff(ps) <= 0):
        raise ValueError("percentiles must be strictly increasing")
    grid = np.linspace(0.0, 1.0, q.cdf_size)
    cdf = np.interp(grid, locs, ps)
    low_slope = (ps[1] - ps[0]) / (locs[1] - locs[0])
    high_slope = (ps[-1] - ps[-2]) / (locs[-1] - locs[-2])
    below = grid < locs[0]
    above = grid > locs[-1]
    cdf[below] = ps[0] + low_slope * (grid[below] - locs[0])
    cdf[above] = ps[-1] + high_slope * (grid[above] - locs[-1])
    cdf = np.clip(cdf, 0.0, 1.0)
    if not q.open_lower_bound and not q.open_upper_bound and cdf[-1] - cdf[0] < 0.01:
        raise ValueError("almost no probability inside a closed range")
    return cdf


def smooth(cdf: np.ndarray) -> np.ndarray:
    """Spread each bin's mass a little onto its neighbours, keeping the total inside the range."""
    pmf = np.diff(cdf)
    if len(pmf) < 3:
        return cdf
    padded = np.concatenate([[pmf[0]], pmf, [pmf[-1]]])
    smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    total = pmf.sum()
    if smoothed.sum() > 0:
        smoothed *= total / smoothed.sum()
    return np.concatenate([[cdf[0]], cdf[0] + np.cumsum(smoothed)])


def standardize(cdf: np.ndarray, q: Question) -> list[float]:
    cdf = np.asarray(cdf, dtype=float).copy()
    n = len(cdf)
    lower_open, upper_open = q.open_lower_bound, q.open_upper_bound
    floor_at = 0.0 if lower_open else cdf[0]
    top_at = 1.0 if upper_open else cdf[-1]
    mass = top_at - floor_at
    if mass <= 1e-9:
        raise ValueError("no probability inside the range")
    location = np.linspace(0.0, 1.0, n)
    rescaled = (cdf - floor_at) / mass
    if lower_open and upper_open:
        cdf = 0.988 * rescaled + 0.01 * location + 0.001
    elif lower_open:
        cdf = 0.989 * rescaled + 0.01 * location + 0.001
    elif upper_open:
        cdf = 0.989 * rescaled + 0.01 * location
    else:
        cdf = 0.99 * rescaled + 0.01 * location

    pmf = np.diff(cdf, prepend=0.0, append=1.0)
    cap = bin_cap(q) * BIN_CAP_MARGIN
    if pmf[1:-1].max() > cap:
        def capped(scale: float) -> np.ndarray:
            return np.concatenate([pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]])

        lo = hi = 1.0
        for _ in range(100):
            if capped(hi).sum() >= 1.0:
                break
            hi *= 1.2
        else:
            raise ValueError("cannot satisfy the bin cap")
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if capped(mid).sum() < 1.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-7:
                break
        pmf = capped(hi)
        pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
        cdf = np.cumsum(pmf)[:-1]
    if not lower_open:
        cdf[0] = 0.0
    if not upper_open:
        cdf[-1] = 1.0
    return [float(x) for x in np.round(cdf, 10)]


def problems(cdf: list[float], q: Question) -> list[str]:
    """What is wrong with a CDF under the Metaculus rules; empty when it is valid."""
    found = []
    arr = np.asarray(cdf, dtype=float)
    n = q.cdf_size
    if len(arr) != n:
        return [f"length {len(arr)} != {n}"]
    if not np.all(np.isfinite(arr)):
        return ["not finite"]
    steps = np.diff(arr)
    if steps.min() < 0.01 / (n - 1) * 0.99:
        found.append("increments too small")
    if steps.max() > bin_cap(q) + 1e-9:
        found.append("a bin above the cap")
    if q.open_lower_bound:
        if arr[0] < 0.001 - 1e-9:
            found.append("too little mass below an open lower bound")
    elif abs(arr[0]) > 1e-9:
        found.append("mass below a closed lower bound")
    if q.open_upper_bound:
        if arr[-1] > 0.999 + 1e-9:
            found.append("too little mass above an open upper bound")
    elif abs(arr[-1] - 1.0) > 1e-9:
        found.append("mass above a closed upper bound")
    return found


def quantile(cdf: list[float], q: Question, p: float) -> float:
    """The value at which the CDF reaches p (clamped to the range)."""
    arr = np.asarray(cdf, dtype=float)
    grid = np.linspace(0.0, 1.0, len(arr))
    location = float(np.interp(p, arr, grid))
    return location_to_value(location, q)
