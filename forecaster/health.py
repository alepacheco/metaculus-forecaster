"""health.json: what the bot's keepers read. Counts, costs and error types only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import __version__, logs

DONE = {"done", "completed", "already_done"}
SKIPPED = {"skipped_existing"}
FAILED = {"failed", "budget"}


class Health:
    def __init__(self, live_dir: Path):
        self.live_dir = live_dir
        self.events_path = live_dir / "events.jsonl"
        self.path = live_dir / "health.json"
        self.last_error_type: str | None = None
        self.last_error_at: str | None = None

    def outcome(self, question_id: int, outcome: str) -> None:
        with self.events_path.open("a") as f:
            f.write(json.dumps({"ts": int(time.time()), "qid": int(question_id), "outcome": outcome}) + "\n")

    def error(self, exc: BaseException) -> None:
        self.last_error_type = type(exc).__name__
        self.last_error_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _last_day(self) -> dict:
        now = int(time.time())
        events = []
        if self.events_path.exists():
            for line in self.events_path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        recent = [e for e in events if now - e["ts"] <= 86400]
        keep = [e for e in events if now - e["ts"] <= 2 * 86400]
        if len(keep) != len(events):
            tmp = self.events_path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(e) + "\n" for e in keep))
            os.replace(tmp, self.events_path)
        latest: dict[int, str] = {}
        posted = set()
        for e in recent:
            latest[e["qid"]] = e["outcome"]
            if e["outcome"] in ("done", "completed"):
                posted.add(e["qid"])
        return {
            "seen": len(latest),
            "done": sum(1 for o in latest.values() if o in DONE),
            "posted": len(posted),
            "skipped": sum(1 for o in latest.values() if o in SKIPPED),
            "failed": sum(1 for o in latest.values() if o in FAILED),
        }

    def write(self, tournaments: list[str], budget, credit_left: float | None,
              pass_seconds: float | None, open_now: int, state: str = "running") -> dict:
        data = {
            "version": __version__,
            "state": state,
            "pass_finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pass_seconds": round(pass_seconds, 1) if pass_seconds is not None else None,
            "tournaments": tournaments,
            "open_now": open_now,
            "last_24h": self._last_day(),
            "spend_usd": round(budget.spent, 4),
            "budget_usd": budget.season_cap,
            "budget_label": budget.label,
            "budget_mode": budget.mode,
            "budget_warning": budget.warning,
            "credit_left_usd": round(credit_left, 4) if credit_left is not None else None,
            "last_error_type": self.last_error_type,
            "last_error_at": self.last_error_at,
            "errors_24h": logs.recent_errors(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, self.path)
        return data
