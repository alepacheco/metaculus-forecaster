"""The bot's loop: a pass every 20 minutes, off the hour, until something needs a person.

Exit codes (the container's keepers act on them):
  0  stopped on purpose
  1  another copy of the bot holds the live directory, or an unexpected crash
  2  a key, credit or API failure that lasted three passes
  3  the season's budget is spent
  4  every question failed three passes in a row (more than one question)
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
from .pipeline import Context, ModelAccessError, forecast_question, models_out

PASS_MINUTES = (7, 27, 47)
PASS_TIMEOUT_SECONDS = 3600
QUESTION_TIMEOUT_SECONDS = 1200
START_DEADLINE_SECONDS = 1800  # no new question starts later than this into a pass
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

    failed_ids: set[int] = field(default_factory=set)

    @property
    def all_failed(self) -> bool:
        return self.attempted > 0 and self.failed == self.attempted


async def run_pass(ctx: Context, tournaments: list[str], health: Health,
                   start_deadline_seconds: float = START_DEADLINE_SECONDS) -> PassResult:
    result = PassResult()
    questions: dict[int, Question] = {}
    for tournament in tournaments:
        try:
            found, unsupported = await ctx.api.open_questions(tournament)
        except (ApiError, httpx.HTTPError, ValueError) as exc:
            logs.error(exc, stage="list")
            result.listing_failed += 1
            continue
        logs.event("listed", tournament=tournament, open=len(found), unsupported=unsupported)
        for q in found:
            questions.setdefault(q.id, q)
    # Every listing failing is a key or site problem; one failing is that tournament's.
    if tournaments and result.listing_failed == len(tournaments):
        result.access_failure = True
    result.open_now = len(questions)
    started = time.monotonic()

    async def one(question: Question) -> str:
        async with ctx.question_semaphore:
            if time.monotonic() - started > start_deadline_seconds:
                return "deferred"  # picked up by the next pass
            try:
                return await asyncio.wait_for(forecast_question(ctx, question), QUESTION_TIMEOUT_SECONDS)
            except BudgetExhausted:
                return "budget"
            except ModelAccessError as exc:
                result.access_failure = True
                logs.error(exc, stage="forecast", question_id=question.id, question_type=question.type)
                return "failed"
            except Exception as exc:  # one question's error never stops the others
                logs.error(exc, stage="forecast", question_id=question.id, question_type=question.type)
                return "failed"

    ordered = list(questions.values())
    outcomes = await asyncio.gather(*[one(q) for q in ordered])
    for question, outcome in zip(ordered, outcomes):
        health.outcome(question.id, outcome)
        result.count(outcome)
        if outcome == "failed":
            result.failed_ids.add(question.id)
        if outcome not in ("already_done", "skipped_existing", "gave_up", "deferred"):
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
    failing_ids: set[int] = set()
    passes = 0
    while True:
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(run_pass(ctx, tournaments, health), PASS_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logs.event("exit", reason="pass_timeout")
            health.write(tournaments, ctx.budget, None, None, 0, state="pass_timeout",
                         models_out=models_out(ctx))
            return EXIT_PASS_TIMEOUT
        elapsed = time.monotonic() - started
        credit = await ctx.models.credit_left()
        usage = await ctx.models.key_usage()
        if usage is not None:
            ctx.budget.reconcile(usage)
        passes += 1

        access_failures = access_failures + 1 if result.access_failure else 0
        # A systemic failure: every question tried failed, three passes running, and more than
        # one question among them (a single odd question is given up on by itself).
        if result.all_failed:
            all_failing += 1
            failing_ids |= result.failed_ids
        else:
            all_failing, failing_ids = 0, set()
        if ctx.budget.mode == "stopped" or result.outcomes.get("budget"):
            health.write(tournaments, ctx.budget, credit, elapsed, result.open_now, state="budget_spent",
                         models_out=models_out(ctx))
            logs.event("exit", reason="budget")
            return EXIT_BUDGET
        if access_failures >= FAILURES_TO_EXIT:
            health.write(tournaments, ctx.budget, credit, elapsed, result.open_now, state="access_failure",
                         models_out=models_out(ctx))
            logs.event("exit", reason="access")
            return EXIT_ACCESS
        if all_failing >= FAILURES_TO_EXIT and len(failing_ids) >= 2:
            health.write(tournaments, ctx.budget, credit, elapsed, result.open_now, state="all_failing",
                         models_out=models_out(ctx))
            logs.event("exit", reason="all_failing")
            return EXIT_ALL_FAILING
        health.write(tournaments, ctx.budget, credit, elapsed, result.open_now, models_out=models_out(ctx))
        if max_passes is not None and passes >= max_passes:
            return 0
        now = datetime.now(timezone.utc)
        await sleep(max(1.0, (next_pass_time(now) - now).total_seconds()))
