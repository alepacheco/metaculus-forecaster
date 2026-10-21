"""Logging that carries counts and error locations only.

The log is read by the bot's keepers while questions are open, so it must never carry a
forecast, a probability, a model's text or an exception message (which can quote a model).
`event` accepts numbers, booleans and short identifiers; anything else is replaced.
`error` records an exception's type, the file and line in this package where it surfaced,
the question id and the question type, never its message; an HTTP failure adds its status code.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

_LOGGER = logging.getLogger("forecaster")
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9_\-./:]{0,64}$")
_PACKAGE_DIR = Path(__file__).resolve().parent
_RECENT: list[tuple[float, str]] = []  # (time, "stage:type[:status]") of each error logged


def setup(log_file: Path | None) -> None:
    _LOGGER.handlers.clear()
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    fmt.converter = __import__("time").gmtime
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    _LOGGER.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(fmt)
        _LOGGER.addHandler(handler)


def _safe(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    if value is None:
        return "-"
    text = str(value)
    return text if _SAFE_TEXT.match(text) else "<redacted>"


def event(name: str, **fields: object) -> None:
    parts = [f"event={_safe(name)}"] + [f"{k}={_safe(v)}" for k, v in fields.items()]
    _LOGGER.info(" ".join(parts))


def error_location(exc: BaseException) -> str:
    """The deepest frame of the traceback that is inside this package, as file:line."""
    where = "unknown"
    for frame in traceback.extract_tb(exc.__traceback__):
        try:
            path = Path(frame.filename).resolve()
        except OSError:
            continue
        if _PACKAGE_DIR in path.parents or path.parent == _PACKAGE_DIR:
            where = f"{path.relative_to(_PACKAGE_DIR.parent)}:{frame.lineno}"
    return where


def error(exc: BaseException, stage: str, question_id: int | None = None, question_type: str | None = None) -> None:
    parts = [
        "event=error",
        f"type={_safe(type(exc).__name__)}",
        f"at={_safe(error_location(exc))}",
        f"stage={_safe(stage)}",
        f"qid={_safe(question_id)}",
        f"qtype={_safe(question_type)}",
    ]
    key = f"{_safe(stage)}:{_safe(type(exc).__name__)}"
    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        parts.append(f"status={status}")
        key += f":{status}"
    _RECENT.append((time.time(), key))
    _LOGGER.warning(" ".join(parts))


def recent_errors(window: float = 86400) -> dict[str, int]:
    """The errors this process logged in the last `window` seconds, counted by stage, type and status."""
    cutoff = time.time() - window
    _RECENT[:] = [e for e in _RECENT if e[0] >= cutoff]
    return dict(Counter(key for _, key in _RECENT))
