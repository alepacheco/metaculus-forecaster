"""The spending ledger and its caps.

Every paid call's cost is added per question and per season. Near the season's cap the bot
switches to a cheaper ensemble and raises a warning; at the cap it makes no more paid calls.
A season budget of zero means no paid calls at all: the bot uses only no-cost models, and
stops if any call is ever charged.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

STEP_DOWN_FRACTION = 0.8


class BudgetExhausted(Exception):
    """The season's budget is spent: no more paid calls."""


class QuestionCapReached(Exception):
    """This question's spending cap is reached."""


class Budget:
    def __init__(self, path: Path | None, label: str, season_cap: float, per_question_cap: float):
        self.path = path
        self.label = label
        self.season_cap = season_cap
        self.per_question_cap = per_question_cap
        self.spent = 0.0
        self.per_question: dict[str, float] = {}
        self._reserved = 0.0
        self._reserved_per_question: dict[str, float] = {}
        self.key_usage_start: float | None = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        season = data.get(self.label, {})
        self.spent = float(season.get("spent_usd", 0.0))
        self.per_question = {k: float(v) for k, v in season.get("per_question_usd", {}).items()}
        self.key_usage_start = season.get("key_usage_start")

    def _save(self) -> None:
        if self.path is None:
            return
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        data[self.label] = {"spent_usd": round(self.spent, 6),
                            "key_usage_start": self.key_usage_start,
                            "per_question_usd": {k: round(v, 6) for k, v in self.per_question.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, self.path)

    def reconcile(self, key_usage: float) -> None:
        """Count at least what the model key itself reports spent since this season began
        (calls that timed out or were cut off are billed but report no cost)."""
        with self._lock:
            if self.key_usage_start is None:
                self.key_usage_start = key_usage - self.spent
                self._save()
                return
            billed = key_usage - self.key_usage_start
            if billed > self.spent:
                self.spent = billed
                self._save()

    @property
    def mode(self) -> str:
        if self.season_cap <= 0:
            return "free" if self.spent <= 0 else "stopped"
        if self.spent >= self.season_cap:
            return "stopped"
        if self.spent >= STEP_DOWN_FRACTION * self.season_cap:
            return "reduced"
        return "full"

    @property
    def warning(self) -> bool:
        return self.mode in ("reduced", "stopped")

    def question_spent(self, question_id: int) -> float:
        return self.per_question.get(str(question_id), 0.0)

    def reserve(self, question_id: int, estimate: float) -> None:
        """Hold `estimate` for a call about to be made; raise unless it fits both caps.

        Calls in flight hold their estimate, so concurrent calls cannot overshoot a cap.
        """
        key = str(question_id)
        with self._lock:
            if self.spent + self._reserved + estimate > self.season_cap:
                raise BudgetExhausted()
            held = self._reserved_per_question.get(key, 0.0)
            if self.question_spent(question_id) + held + estimate > self.per_question_cap:
                raise QuestionCapReached()
            self._reserved += estimate
            self._reserved_per_question[key] = held + estimate

    def settle(self, question_id: int, estimate: float, cost: float) -> None:
        """Release a reservation and charge what the call actually cost."""
        key = str(question_id)
        with self._lock:
            self._reserved = max(0.0, self._reserved - estimate)
            self._reserved_per_question[key] = max(0.0, self._reserved_per_question.get(key, 0.0) - estimate)
            if cost:
                self.spent += cost
                self.per_question[key] = self.per_question.get(key, 0.0) + cost
                self._save()
