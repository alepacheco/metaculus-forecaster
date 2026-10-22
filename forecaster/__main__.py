"""Command line.

  python -m forecaster run                 forecast every 20 minutes until stopped (the live bot)
  python -m forecaster pass [--tournament SLUG ...] [--allow-update]
                                           one pass now (--allow-update only in the testing area)
  python -m forecaster dry-run --tournament SLUG --out DIR [--per-type N] [--budget USD] [--backtest]
                             [--no-news]   forecast closed or resolved questions without posting
  python -m forecaster analyse [--records DIR]
                                           scores on resolved questions
  python -m forecaster check [--models]    are the keys working (prints ok or failed per key), and
                                           does each no-cost model answer a short neutral request
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import sys
from pathlib import Path

from . import logs
from .analyse import analyse
from .budget import Budget
from .config import load_config, load_secrets
from .health import Health
from .llm import Member, ModelClient, ModelError
from .metaculus import MetaculusClient, Question
from .pipeline import FREE, FREE_RESERVE, Context, forecast_question, models_out
from .records import Records
from .research import NewsClient
from .runner import run_forever, run_pass

TESTING_AREA = "bot-testing-area"


def _clients(secrets: dict[str, str], budget: Budget):
    api = MetaculusClient(secrets["METACULUS_TOKEN"])
    models = ModelClient(secrets.get("OPENROUTER_API_KEY", ""), budget)
    news = NewsClient(secrets["ASKNEWS_API_KEY"]) if secrets.get("ASKNEWS_API_KEY") else None
    return api, models, news


async def _close(*clients) -> None:
    for c in clients:
        if c is not None:
            await c.close()


def _hold_lock(live_dir: Path):
    """One copy of the bot per live directory: a second would double its spend and its posts."""
    handle = open(live_dir / ".lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


async def cmd_run(args) -> int:
    config = load_config()
    logs.setup(config.live_dir / "bot.log")
    tournaments = config.tournaments
    if args.command == "pass":
        tournaments = args.tournament or config.tournaments
        if any(t not in config.tournaments and t != TESTING_AREA for t in tournaments):
            print("a pass takes only the config's tournaments or the testing area", file=sys.stderr)
            return 1
        if args.allow_update and tournaments != [TESTING_AREA]:
            print("--allow-update is only for the testing area", file=sys.stderr)
            return 1
    lock = _hold_lock(config.live_dir)
    if lock is None:
        logs.event("exit", reason="already_running")
        return 1
    budget = Budget(config.live_dir / "ledger.json", config.budget_label,
                    config.season_budget_usd, config.per_question_cap_usd)
    api, models, news = _clients(config.secrets, budget)
    ctx = Context(api=api, models=models, news=news, budget=budget, records=Records(config.records_dir))
    health = Health(config.live_dir)
    logs.event("start", tournaments=len(config.tournaments), budget_mode=budget.mode)
    try:
        if args.command == "pass":
            ctx.allow_update = args.allow_update
            result = await run_pass(ctx, tournaments, health)
            health.write(tournaments, budget, await models.credit_left(), None, result.open_now,
                         models_out=models_out(ctx))
            return 0 if not result.failed else 1
        return await run_forever(ctx, config.tournaments, health)
    finally:
        await _close(api, models, news)
        lock.close()


async def cmd_dry_run(args) -> int:
    secrets = load_secrets(Path(args.secret_file))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs.setup(out / "dry-run.log")
    budget = Budget(out / "ledger.json", "dry-run", args.budget, args.per_question_cap)
    if args.no_news:
        secrets.pop("ASKNEWS_API_KEY", None)
    api, models, news = _clients(secrets, budget)
    ctx = Context(api=api, models=models, news=news, budget=budget, records=Records(out / "records"),
                  post=False, backtest=args.backtest)
    try:
        if "open" in args.statuses.split(",") and args.tournament != TESTING_AREA:
            print("dry runs use closed or resolved questions only", file=sys.stderr)
            return 1
        posts = await api.list_posts(args.tournament, statuses=args.statuses, max_posts=args.max_posts,
                                     order_by="-actual_close_time" if "open" not in args.statuses else "-open_time")
        chosen: list[Question] = []
        per_type: dict[str, int] = {}
        for post in posts:
            q = Question.from_post(post, args.tournament)
            if q is None or (args.tournament != TESTING_AREA and q.status not in ("closed", "resolved")):
                continue
            if per_type.get(q.type, 0) >= args.per_type:
                continue
            per_type[q.type] = per_type.get(q.type, 0) + 1
            chosen.append(q)
        logs.event("dry_run_selected", questions=len(chosen), **{k: v for k, v in sorted(per_type.items())})
        outcomes: dict[str, int] = {}

        async def one(q: Question) -> str:
            async with ctx.question_semaphore:
                try:
                    return await forecast_question(ctx, q)
                except Exception as exc:
                    logs.error(exc, stage="dry_run", question_id=q.id, question_type=q.type)
                    return "failed"

        for outcome in await asyncio.gather(*[one(q) for q in chosen]):
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        logs.event("dry_run_done", spent_usd=budget.spent, **outcomes)
        return 0 if not outcomes.get("failed") else 1
    finally:
        await _close(api, models, news)


async def cmd_analyse(args) -> int:
    secrets = load_secrets(Path(args.secret_file))
    records_dir = Path(args.records) if args.records else load_config().records_dir
    api = MetaculusClient(secrets["METACULUS_TOKEN"])
    try:
        print(json.dumps(await analyse(api, Records(records_dir)), indent=1))
    finally:
        await api.close()
    return 0


async def cmd_check(args) -> int:
    secrets = load_secrets(Path(args.secret_file))
    budget = Budget(None, "check", 0.0, 0.0)
    api, models, news = _clients(secrets, budget)
    status = {}
    try:
        try:
            me = await api.me()
            status["metaculus"] = "ok" if me.get("is_bot") else "ok-not-a-bot-account"
        except Exception as exc:
            status["metaculus"] = f"failed:{type(exc).__name__}"
        credit = await models.credit_left()
        status["openrouter"] = "ok" if credit is not None else "failed-or-no-limit"
        status["openrouter_credit_usd"] = credit
        if news is None:
            status["asknews"] = "no-key"
        else:
            try:
                await news.search("global news", 1, "latest news")
                status["asknews"] = "ok"
            except Exception as exc:
                status["asknews"] = f"failed:{type(exc).__name__}"
        if args.models:
            # Each no-cost model, asked something that is no question: does it answer at all?
            listed = await models.listed_models()
            probes = {}
            for model in dict.fromkeys(m.model for m in FREE + FREE_RESERVE):
                probes[model] = await _probe(models, model, listed)
            status["models"] = probes
    finally:
        await _close(api, models, news)
    print(json.dumps(status))
    return 0 if all(v == "ok" or k in ("openrouter_credit_usd", "models") for k, v in status.items()) else 1


async def _probe(models: ModelClient, model: str, listed: set[str] | None) -> str:
    if listed is not None and model not in listed:
        return "unlisted"
    try:
        await models.complete(0, Member(model, "low", max_tokens=4000, estimate_usd=0.0),
                              "Reply with the single word: ready")
        return "ok"
    except ModelError as exc:
        return f"failed:{exc.status}" + (f":{exc.reason}" if exc.reason else "")
    except Exception as exc:
        return f"failed:{type(exc).__name__}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecaster")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    p = sub.add_parser("pass")
    p.add_argument("--tournament", action="append")
    p.add_argument("--allow-update", action="store_true")
    d = sub.add_parser("dry-run")
    d.add_argument("--tournament", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--statuses", default="resolved")
    d.add_argument("--per-type", type=int, default=2)
    d.add_argument("--max-posts", type=int, default=300)
    d.add_argument("--budget", type=float, default=5.0)
    d.add_argument("--per-question-cap", type=float, default=2.0)
    d.add_argument("--backtest", action="store_true")
    d.add_argument("--no-news", action="store_true")
    for name in ("dry-run", "analyse", "check"):
        s = d if name == "dry-run" else sub.add_parser(name)
        s.add_argument("--secret-file", default="/secret/bot.env")
        if name == "analyse":
            s.add_argument("--records")
        if name == "check":
            s.add_argument("--models", action="store_true")
    args = parser.parse_args(argv)
    command = {"run": cmd_run, "pass": cmd_run, "dry-run": cmd_dry_run,
               "analyse": cmd_analyse, "check": cmd_check}[args.command]
    try:
        return asyncio.run(command(args))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # the log gets the error's type and place, never a traceback's text
        logs.error(exc, stage="main")
        return 1


if __name__ == "__main__":
    sys.exit(main())
