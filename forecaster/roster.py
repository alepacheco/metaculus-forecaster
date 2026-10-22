"""Which models can be called now, so a model the provider withdraws mid-season does not
quietly shrink the ensemble.

A model is out while the provider's public list lacks it (or its listed expiry has passed),
and for a day after a call to it answered "not found". Each member whose model is out is
replaced by the first model of its ensemble's reserve that is in, else by the members that
are in, in turn, so the ensemble keeps its number of forecasts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterable

from . import logs
from .llm import Member

REFRESH_SECONDS = 3600  # the provider's list is read at most this often
NOT_FOUND_SECONDS = 86400  # a model that answered "not found" is tried again after this


class Roster:
    def __init__(self) -> None:
        self.listed: set[str] | None = None  # None until the list has been read once
        self.refreshed_at = 0.0
        self.not_found: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def refresh(self, models) -> None:
        """Read the provider's list, if it is due; a list that cannot be read changes nothing."""
        async with self._lock:
            if time.time() - self.refreshed_at < REFRESH_SECONDS:
                return
            self.refreshed_at = time.time()
            listed = await models.listed_models()
            if listed:
                self.listed = listed

    def mark_not_found(self, model: str) -> None:
        if self.is_in(model):
            logs.event("model_out", model=model, reason="not_found")
        self.not_found[model] = time.time()

    def is_in(self, model: str) -> bool:
        if self.listed is not None and model not in self.listed:
            return False
        at = self.not_found.get(model)
        return at is None or time.time() - at >= NOT_FOUND_SECONDS

    def out(self, models: Iterable[str]) -> list[str]:
        return sorted({m for m in models if not self.is_in(m)})

    def ensemble(self, base: list[Member], reserve: Iterable[Member] = ()) -> list[Member]:
        """`base` with each member whose model is out replaced; `base` as written when none of
        its models or its reserve's is known to be in (its calls then fail loudly)."""
        spare = [m for m in reserve if self.is_in(m.model)]
        chosen = [m if self.is_in(m.model) else (spare.pop(0) if spare else None) for m in base]
        filled = [m for m in chosen if m is not None]
        if not filled:
            return list(base)
        return [m if m is not None else filled[i % len(filled)] for i, m in enumerate(chosen)]
