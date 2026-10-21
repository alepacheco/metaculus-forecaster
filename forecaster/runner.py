"""The bot's loop: a pass every 20 minutes, off the hour, until something needs a person.

Exit codes (the container's keepers act on them):
  0  stopped on purpose
  2  a key, credit or API failure that lasted three passes
  3  the season's budget is spent
  4  every question failed three passes in a row
  5  a pass did not finish within an hour
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from . import logs
from .budget import BudgetExhausted
from .health import Health
from .metaculus import ApiError, Question
from .pipeline import Context, ModelAccessError, forecast_question

PASS_MINUTES = (7, 27, 47)
PASS_TIMEOUT_SECONDS = 3600
FAILURES_TO_EXIT = 3

EXIT_ACCESS = 2
EXIT_BUDGET = 3
EXIT_ALL_FAILING = 4
EXIT_PASS_TIMEOUT = 5


@dataclass
class PassResult:
    open_now: int = 0
    attempted: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    access_failure: bool = False
    listing_failed: int = 0

    def count(self, outcome: str) -> None:
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

    @property
    def failed(self) -> int:
        return self.outcomes.get("failed", 0)

    @property
    def all_failed(self) -> bool:
        return self.attempted > 0 and self.failed == self.attempted


async def run_pass(ctx: Context, tournaments: list[str], health: Health) -> PassResult:
    result = PassResult()
    questions: dict[int, Question] = {}
    for tournament in tournaments:
        try:
            found, unsupported = await ctx.api.open_questions(tournament)
        except (ApiError, httpx.HTTPError, ValueError) as exc:
            logs.error(exc, stage="list")
            health.error(exc)
            result.listing_failed += 1
            if isinstance(exc, ApiError) and exc.is_auth:
                result.access_failure = True
            continue
        logs.event("listed", tournament=tournament, open=len(found), unsupported=unsupported)
        for q in found:
            questions.setdefault(q.id, q)
    if tournaments and result.listing_failed == len(tournaments):
        result.access_failure = True
    result.open_now = len(questions)

    async def one(question: Question) -> str:
        async with ctx.question_semaphore:
            try:
                return await forecast_question(ctx, question)
            except BudgetExhausted:
                return "budget"
            except ModelAccessError as exc:
                result.access_failure = True
                logs.error(exc, stage="forecast", question_id=question.id, question_type=question.type)
                health.error(exc)
                return "failed"
            except Exception as exc:  # one question's error never stops the others
                if isinstance(exc, ApiError) and exc.is_auth:
                    result.access_failure = True
                logs.error(exc, stage="forecast", question_id=question.id, question_type=question.type)
                health.error(exc)
                return "failed"

    ordered = list(questions.values())
    outcomes = await asyncio.gather(*[one(q) for q in ordered])
    for question, outcome in zip(ordered, outcomes):
        health.outcome(question.id, outcome)
        result.count(outcome)
        if outcome not in ("already_done", "skipped_existing"):
            result.attempted += 1
    logs.event("pass", open=result.open_now, attempted=result.attempted,
               **{k: v for k, v in sorted(result.outcomes.items())})
    return result


def next_pass_time(now: datetime) -> datetime:
    """The next of minutes 7, 27 and 47 of an hour, strictly after now."""
    base = now.replace(second=0, microsecond=0)
    for add_hour in (0, 1):
        hour_start = base.replace(minute=0) + timedelta(hours=add_hour)
        for minute in PASS_MINUTES:
            candidate = hour_start + timedelta(minutes=minute)
            if candidate > now:
                return candidate
    return base + timedelta(minutes=20)


async def run_forever(ctx: Context, tournaments: list[str], health: Health,
                      max_passes: int | None = None, sleep=asyncio.sleep) -> int:
    access_failures = 0
    all_failing = 0
    passes = 0
    while True:
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(run_pass(ctx, tournaments, health), PASS_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logs.event("exit", reason="pass_timeout")
            health.write(tournaments, ctx.budget, None, None, 0, state="pass_timeout")
            return EXIT_PASS_TIMEOUT
        elapsed = time.monotonic() - started
        credit = await ctx.models.credit_left()
        passes += 1

        access_failures = access_failures + 1 if result.access_failure else 0
        all_failing = all_failing + 1 if result.all_failed else 0
        if ctx.budget.mode == "stopped" or result.outcomes.get("budget"):
            health.write(tournaments, ctx.budget, credit, elapsed, result.open_now, state="budget_spent")
            logs.event("exit", reason="budget")
            return EXIT_BUDGET
        if access_failures >= FAILURES_TO_EXIT:
            health.write(tournaments, ctx.budget, credit, elapsed, result.open_now, state="access_failure")
            logs.event("exit", reason="access")
            return EXIT_ACCESS
        if all_failing >= FAILURES_TO_EXIT:
            health.write(tournaments, ctx.budget, credit, elapsed, result.open_now, state="all_failing")
            logs.event("exit", reason="all_failing")
            return EXIT_ALL_FAILING
        health.write(tournaments, ctx.budget, credit, elapsed, result.open_now)
        if max_passes is not None and passes >= max_passes:
            return 0
        now = datetime.now(timezone.utc)
        await sleep(max(1.0, (next_pass_time(now) - now).total_seconds()))
