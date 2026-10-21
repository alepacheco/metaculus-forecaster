"""Reading a forecast out of a model's reply."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from .metaculus import Question
from .prompts import PERCENTILES


class ParseError(ValueError):
    pass


_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clean(text: str) -> str:
    return text.replace("−", "-").replace("–", "-").replace("*", "").replace("`", "")


def parse_binary(text: str) -> float:
    matches = re.findall(r"Probability\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", _clean(text), re.IGNORECASE)
    if not matches:
        raise ParseError("no probability")
    value = float(matches[-1]) / 100.0
    if not 0.0 <= value <= 1.0:
        raise ParseError("probability out of range")
    return min(max(value, 0.001), 0.999)


def parse_multiple_choice(text: str, options: list[str]) -> list[float]:
    cleaned = _clean(text)
    values = []
    for option in options:
        label = re.escape(_clean(option).strip())
        pattern = rf"^\s*(?:[-•]\s*)?{label}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%"
        matches = re.findall(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
        if not matches:
            raise ParseError("an option has no probability")
        values.append(float(matches[-1]))
    total = sum(values)
    if total <= 0:
        raise ParseError("probabilities sum to zero")
    return [v / total for v in values]


def _parse_value(raw: str, q: Question) -> float:
    raw = raw.strip()
    if q.type == "date":
        match = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        if not match:
            raise ParseError("no date")
        moment = datetime.strptime(match.group(0), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return moment.timestamp()
    match = _NUMBER.search(raw.replace(",", "").replace(" ", ""))
    if not match:
        raise ParseError("no number")
    return float(match.group(0))


def parse_percentiles(text: str, q: Question) -> list[tuple[float, float]]:
    """(percentile in 0..1, value) points, strictly increasing in both."""
    found: dict[int, float] = {}
    for match in re.finditer(r"Percentile\s*(\d{1,2})\s*:\s*([^\n]+)", _clean(text), re.IGNORECASE):
        level = int(match.group(1))
        if level in PERCENTILES:
            found[level] = _parse_value(match.group(2), q)
    if len(found) < 5 or min(found) > 20 or max(found) < 80:
        raise ParseError("too few percentiles")
    levels = sorted(found)
    values = sorted(found[k] for k in levels)  # a reply with values out of order is read in order
    span = (q.range_max - q.range_min) if q.range_max is not None and q.range_min is not None else 1.0
    epsilon = abs(span) * 1e-6 or 1e-6
    points = []
    previous = None
    for level, value in zip(levels, values):
        if previous is not None and value <= previous:
            value = previous + epsilon
        points.append((level / 100.0, value))
        previous = value
    return points


def summary(text: str) -> str:
    """The reply's 'Summary:' passage, without the final-answer lines."""
    cleaned = _clean(text)
    index = cleaned.lower().rfind("summary:")
    if index < 0:
        return ""
    lines = []
    for line in cleaned[index + len("summary:"):].splitlines():
        stripped = line.strip()
        if re.match(r"^(Probability|Percentile)\b", stripped, re.IGNORECASE) or re.search(r":\s*[0-9.]+\s*%\s*$", stripped):
            break
        if not stripped and lines:
            break
        if stripped:
            lines.append(stripped)
    return " ".join(lines).strip()
