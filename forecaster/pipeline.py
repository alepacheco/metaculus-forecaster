"""Forecasting one question: research, an ensemble of forecasters, aggregation, posting."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import aggregate, logs
from . import cdf as cdfs
from .budget import Budget, BudgetExhausted, QuestionCapReached
from .llm import Member, ModelClient, ModelError
from .metaculus import MetaculusClient, Question
from .parse import parse_binary, parse_multiple_choice, parse_percentiles, summary
from .prompts import forecast_prompt
from .records import Records
from .research import NewsClient, gather

# The ensembles. Forecasters from several labs, each thinking hard, five forecasts a question.
MAIN = [
    Member("openai/gpt-6.1-sol", "high", estimate_usd=0.20),
    Member("anthropic/claude-sonnet-5.5", "high", estimate_usd=0.20),
    Member("google/gemini-4-argon", "high", estimate_usd=0.20),
    Member("x-ai/grok-4.7", "high", estimate_usd=0.15),
    Member("openai/gpt-6.1-sol", "high", estimate_usd=0.20),
]
LIGHT = [
    Member("openai/gpt-6-luna", "high", estimate_usd=0.03),
    Member("anthropic/claude-haiku-5.5", "high", estimate_usd=0.03),
    Member("google/gemini-3.8-flash", "high", estimate_usd=0.05),
    Member("deepseek/deepseek-v4.1-flash", "high", estimate_usd=0.02),
    Member("openai/gpt-6-luna", "high", estimate_usd=0.03),
]
# With no season budget: models offered at no cost, from three labs, the strongest twice;
# research is news search only (a web-search brief is a paid call).
FREE = [
    Member("nvidia/nemotron-3-ultra-550b-a55b:free", "high", estimate_usd=0.0),
    Member("minimax/minimax-m3.1-flash:free", "high", estimate_usd=0.0),
    Member("dots-studio/dots-3-note-preview:free", "high", estimate_usd=0.0),
    Member("nvidia/nemotron-3-ultra-550b-a55b:free", "high", estimate_usd=0.0),
    Member("minimax/minimax-m3.1-flash:free", "high", estimate_usd=0.0),
]
assert all(m.model.endswith(":free") and m.estimate_usd == 0 for m in FREE)
WEB_RESEARCHER = Member("openai/gpt-6.1-sol", "medium", max_tokens=12000, estimate_usd=0.10)
MIN_FORECASTS = 2
MAX_ATTEMPTS = 3  # tries at forecasting (or at posting) one question before giving it up

# Comments carry the forecast and its reasons, and name no model, vendor or AI.
_UNSAID = re.compile(
    r"\b(gpt\S*|claude|gemini|gemma|grok|deepseek|llama|mistral|qwen|kimi|nemotron|nvidia|minimax|inkling|"
    r"thinking machines|openai|anthropic|google|deepmind|"
    r"xai|x\.ai|openrouter|chatgpt|llms?|language models?|ai|a\.i\.|artificial intelligence|"
    r"machine learning|models?|chatbot|assistant)\b",
    re.IGNORECASE,
)
MAX_COMMENT_CHARS = 1200


@dataclass
class Context:
    api: MetaculusClient
    models: ModelClient
    news: NewsClient | None
    budget: Budget
    records: Records
    post: bool = True
    backtest: bool = False
    allow_update: bool = False
    question_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))


def choose_ensemble(question: Question, budget: Budget) -> tuple[str, list[Member], Member | None]:
    if budget.mode == "free":
        return "free", FREE, None
    if budget.mode != "full" or question.tournament.startswith("minibench"):
        return "light", LIGHT, None
    return "main", MAIN, WEB_RESEARCHER


def clean_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if s and not _UNSAID.search(s)]
    out = " ".join(kept).strip()
    if len(out) > MAX_COMMENT_CHARS:
        out = out[:MAX_COMMENT_CHARS].rsplit(" ", 1)[0] + "..."
    return out


def _fmt_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _fmt_value(value: float, q: Question) -> str:
    if q.type == "date":
        return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d")
    step = q.discrete_step
    if step:
        decimals = 0 if step >= 1 else max(0, -math.floor(math.log10(step) + 1e-9))
        return f"{round(value / step) * step:,.{decimals}f}"
    return _fmt_number(value)


def forecast_line(question: Question, value) -> str:
    if question.type == "binary":
        return f"Forecast: {value * 100:.1f}% chance of Yes."
    if question.type == "multiple_choice":
        parts = [f"{option}: {value[option] * 100:.1f}%" for option in question.options]
        return "Forecast: " + "; ".join(parts) + "."
    unit = f" {question.unit}" if question.unit and question.type != "date" else ""
    median = _fmt_value(cdfs.quantile(value, question, 0.5), question)
    low = _fmt_value(cdfs.quantile(value, question, 0.1), question)
    high = _fmt_value(cdfs.quantile(value, question, 0.9), question)
    return f"Forecast: median {median}{unit}; 80% interval {low} to {high}{unit}."


def comment_text(question: Question, value, reasons: str) -> str:
    reasons = clean_sentences(reasons)
    if not reasons:
        reasons = "Reasoning: weighed base rates for similar events against the latest news on this question."
    return f"{forecast_line(question, value)}\n\n{reasons}"


async def _member(ctx: Context, question: Question, member: Member, prompt: str) -> dict:
    entry: dict = {"model": member.model, "effort": member.effort}
    try:
        reply = await ctx.models.complete(question.id, member, prompt)
    except (BudgetExhausted, QuestionCapReached) as exc:
        entry["error_type"] = type(exc).__name__
        return entry
    except Exception as exc:
        logs.error(exc, stage="member", question_id=question.id, question_type=question.type)
        entry["error_type"] = type(exc).__name__
        entry["auth"] = bool(isinstance(exc, ModelError) and (exc.is_auth or exc.is_credit))
        return entry
    entry["text"] = reply.text
    entry["cost_usd"] = reply.cost
    try:
        if question.type == "binary":
            entry["forecast"] = parse_binary(reply.text)
        elif question.type == "multiple_choice":
            entry["forecast"] = parse_multiple_choice(reply.text, question.options)
        else:
            entry["forecast"] = parse_percentiles(reply.text, question)
            entry["cdf"] = cdfs.percentiles_to_cdf(entry["forecast"], question).tolist()
    except Exception as exc:  # an unusable reply drops this forecaster only
        logs.error(exc, stage="parse", question_id=question.id, question_type=question.type)
        entry["error_type"] = type(exc).__name__
        entry.pop("forecast", None)
    return entry


def _aggregate(question: Question, members: list[dict]):
    valid = [m for m in members if "forecast" in m]
    if question.type == "binary":
        value = aggregate.binary([m["forecast"] for m in valid])
        closest = min(valid, key=lambda m: abs(m["forecast"] - value))
        return value, closest
    if question.type == "multiple_choice":
        probs = aggregate.multiple_choice([m["forecast"] for m in valid])
        value = dict(zip(question.options, probs))
        closest = min(valid, key=lambda m: sum(abs(a - b) for a, b in zip(m["forecast"], probs)))
        return value, closest
    member_cdfs = [np.asarray(m["cdf"]) for m in valid]
    value = aggregate.continuous(member_cdfs, question)
    target = float(np.interp(0.5, value, np.linspace(0, 1, len(value))))
    closest = min(valid, key=lambda m: abs(float(np.interp(0.5, m["cdf"], np.linspace(0, 1, len(m["cdf"])))) - target))
    return value, closest


class NoForecast(Exception):
    """Too few of the ensemble's forecasts could be used."""


class ModelAccessError(Exception):
    """The model provider refused every call: a key or credit problem."""


async def _finish(ctx: Context, question: Question, record: dict) -> None:
    """Post what the record holds that is not posted yet: the forecast, then the comment."""
    if not record.get("forecast_posted_at"):
        if not ctx.allow_update and await ctx.api.has_my_forecast(question.post_id):
            record["forecast_posted_at"] = "found-on-site"
        else:
            await ctx.api.post_forecast(question, record["value"])
            record["forecast_posted_at"] = datetime.now(timezone.utc).isoformat()
        ctx.records.save(question.id, record)
    if not record.get("comment_posted_at"):
        await ctx.api.post_comment(question.post_id, record["comment"])
        record["comment_posted_at"] = datetime.now(timezone.utc).isoformat()
        ctx.records.save(question.id, record)


async def forecast_question(ctx: Context, question: Question) -> str:
    """Forecast one question, post it with its comment. Returns the outcome."""
    record = ctx.records.load(question.id)
    if record and record.get("value") is not None:
        if record.get("comment_posted_at") or not ctx.post:
            return "already_done"
        if record.get("finish_attempts", 0) >= MAX_ATTEMPTS:
            return "gave_up"
        record["finish_attempts"] = record.get("finish_attempts", 0) + 1
        ctx.records.save(question.id, record)
        await _finish(ctx, question, record)
        return "completed"
    attempts = (record or {}).get("attempts", 0)
    if attempts >= MAX_ATTEMPTS:
        return "gave_up"
    if not ctx.allow_update and ctx.post:
        # The question list does not say whether we forecast a question; its page does.
        if question.has_my_forecast or await ctx.api.has_my_forecast(question.post_id):
            return "skipped_existing"
    if ctx.budget.mode == "stopped":
        return "budget"

    ensemble_name, ensemble, researcher = choose_ensemble(question, ctx.budget)
    record = {
        "question": {"id": question.id, "post_id": question.post_id, "type": question.type,
                     "title": question.title, "tournament": question.tournament,
                     "open_time": question.open_time, "scheduled_close_time": question.scheduled_close_time},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts + 1,
        "ensemble": ensemble_name,
        "backtest": ctx.backtest,
        "value": None,
    }
    # Saved before any paid call, so a question that keeps failing is given up after a few tries.
    ctx.records.save(question.id, record)
    as_of = question.open_timestamp() if ctx.backtest else None
    try:
        research = await gather(question, ctx.models, ctx.news, researcher, as_of=as_of,
                                background=not question.tournament.startswith("minibench"))
    except BudgetExhausted:
        return "budget"
    today = datetime.fromtimestamp(as_of, timezone.utc) if as_of else datetime.now(timezone.utc)
    prompt = forecast_prompt(question, research, today.strftime("%Y-%m-%d"))
    members = await asyncio.gather(*[_member(ctx, question, m, prompt) for m in ensemble])
    valid = [m for m in members if "forecast" in m]
    record.update({"research": research, "members": members,
                   "cost_usd": round(ctx.budget.question_spent(question.id), 6)})
    if len(valid) < MIN_FORECASTS:
        ctx.records.save(question.id, record)
        if all(m.get("auth") for m in members):
            raise ModelAccessError()
        if any(m.get("error_type") == "BudgetExhausted" for m in members):
            return "budget"
        raise NoForecast()
    try:
        value, closest = _aggregate(question, valid)
    except Exception:
        ctx.records.save(question.id, record)
        raise NoForecast()
    record["value"] = value
    record["comment"] = comment_text(question, value, summary(closest.get("text", "")))
    ctx.records.save(question.id, record)
    if not ctx.post:
        return "done"
    await _finish(ctx, question, record)
    return "done"
