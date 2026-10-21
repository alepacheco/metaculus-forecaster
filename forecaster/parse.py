"""Reading a forecast out of a model's reply.

A reply that cannot be read exactly is refused rather than guessed at: the forecaster is
dropped from the ensemble, and the others carry the forecast.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from . import cdf
from .metaculus import Question
from .prompts import PERCENTILES


class ParseError(ValueError):
    pass


_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# A number written with a magnitude word or suffix ("5k", "7.2 trillion", "3 x 10^6") is refused:
# the prompt asks for plain numbers in the question's units, and a guess at the scale can be
# off by a factor of a million.
_MAGNITUDE = re.compile(
    rf"({_NUMBER})\s*(?:(thousand|million|billion|trillion|bn|mn|k|m|b|t)(?![a-z])|[x×*]\s*10\s*\^)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return (text.replace("−", "-").replace("–", "-").replace("—", "-")
            .replace("*", "").replace("`", "").replace(" ", " "))


def parse_binary(text: str) -> float:
    pattern = rf"Probability[^:\n]{{0,24}}:\s*~?\s*({_NUMBER})\s*%"
    matches = re.findall(pattern, _clean(text), re.IGNORECASE)
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
        pattern = (rf"^\s*(?:[-•]\s*)?(?:(?:\d+|[A-Za-z])[.)]\s+)?[\"'“”]?{label}[\"'“”]?"
                   rf"\s*[:=-]\s*~?\s*({_NUMBER})\s*%")
        matches = re.findall(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
        if not matches:
            raise ParseError("an option has no probability")
        values.append(float(matches[-1]))
    if any(v < 0 for v in values):
        raise ParseError("negative probability")
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
    plain = re.sub(r"(?<=\d),(?=\d{3}\b)", "", raw).replace("$", "").replace("€", "").replace("£", "")
    if _MAGNITUDE.search(plain):
        raise ParseError("number with a magnitude word")
    match = re.search(_NUMBER, plain)
    if not match:
        raise ParseError("no number")
    return float(match.group(0))


_PERCENTILE_LINE = re.compile(
    r"^\s*(?:[-•]\s*)?(?:Percentile\s*(\d{1,2})|P(\d{1,2})|(\d{1,2})(?:st|nd|rd|th)\s+percentile)\s*[:=]\s*([^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_percentiles(text: str, q: Question) -> list[tuple[float, float]]:
    """(percentile in 0..1, value) points, strictly increasing in both."""
    found: dict[int, float] = {}
    for match in _PERCENTILE_LINE.finditer(_clean(text)):
        level = int(match.group(1) or match.group(2) or match.group(3))
        if level in PERCENTILES:
            found[level] = _parse_value(match.group(4), q)
    if len(found) < 5 or min(found) > 20 or max(found) < 80:
        raise ParseError("too few percentiles")
    levels = sorted(found)
    values = [found[k] for k in levels]
    if any(b < a for a, b in zip(values, values[1:])):
        raise ParseError("values go down")
    # Equal neighbours are separated by a hair in the question's scaled location, so the
    # nudge stays tiny on a logarithmic range too.
    points = []
    previous_location = None
    for level, value in zip(levels, values):
        location = cdf.value_to_location(value, q)
        if previous_location is not None and location <= previous_location:
            location = previous_location + 1e-6
            value = cdf.location_to_value(location, q)
        points.append((level / 100.0, value))
        previous_location = location
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
        if re.match(r"^(Probability|Percentile|P\d{1,2}\s*:)", stripped, re.IGNORECASE) or re.search(r"[:=-]\s*~?\s*[0-9.]+\s*%\s*$", stripped):
            break
        if not stripped and lines:
            break
        if stripped:
            lines.append(stripped)
    return " ".join(lines).strip()
