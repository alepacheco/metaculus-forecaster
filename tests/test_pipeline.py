import asyncio
import re

import pytest

from fakes import FakeApi, FakeModels, binary_q, choice_q, date_q, discrete_q, log_q, numeric_q, reply_for
from forecaster import logs, pipeline
from forecaster.budget import Budget
from forecaster.health import Health
from forecaster.pipeline import Context, forecast_question
from forecaster.records import Records
from forecaster.runner import EXIT_ALL_FAILING, EXIT_BUDGET, run_forever, run_pass

ALL = [binary_q, choice_q, numeric_q, log_q, discrete_q, date_q]
UNSAID = re.compile(r"\b(gpt\S*|claude|gemini|grok|openai|anthropic|google|llm|language model|ai|model)\b", re.I)


def make_ctx(tmp_path, questions, budget_cap=100.0, per_question=5.0, replies=None, on_site=(), post=True,
             fail_models=()):
    budget = Budget(tmp_path / "ledger.json", "test", budget_cap, per_question)
    models = FakeModels(budget, replies=replies, fail_models=fail_models)
    models.questions = {q.id: q for q in questions}
    api = FakeApi(questions, on_site=on_site)
    ctx = Context(api=api, models=models, news=None, budget=budget, records=Records(tmp_path / "records"), post=post)
    return ctx, api, models


@pytest.mark.parametrize("make", ALL)
def test_every_type_posts_a_forecast_then_its_comment(tmp_path, make):
    q = make()
    ctx, api, _ = make_ctx(tmp_path, [q])
    assert asyncio.run(forecast_question(ctx, q)) == "done"
    assert [f[0] for f in api.forecasts] == [q.id]
    assert len(api.comments) == 1 and api.comments[0][0] == q.post_id
    assert api.comments[0][1].startswith("Forecast:")
    record = ctx.records.load(q.id)
    assert record["forecast_posted_at"] and record["comment_posted_at"]


def test_a_done_question_is_skipped(tmp_path):
    q = binary_q()
    ctx, api, models = make_ctx(tmp_path, [q])
    asyncio.run(forecast_question(ctx, q))
    calls = len(models.calls)
    assert asyncio.run(forecast_question(ctx, q)) == "already_done"
    assert len(models.calls) == calls and len(api.forecasts) == 1 and len(api.comments) == 1


def test_a_forecast_on_the_site_without_a_record_is_never_redone(tmp_path):
    q = binary_q()  # the question list does not show it; the question's page does
    ctx, api, models = make_ctx(tmp_path, [q], on_site=[q.id])
    assert asyncio.run(forecast_question(ctx, q)) == "skipped_existing"
    assert not models.calls and not api.forecasts and not api.comments


def test_a_half_done_question_is_finished_from_its_record(tmp_path):
    q = numeric_q()
    ctx, api, models = make_ctx(tmp_path, [q], post=False)
    asyncio.run(forecast_question(ctx, q))  # forecast made and recorded, nothing posted
    recorded = ctx.records.load(q.id)["value"]
    calls = len(models.calls)
    ctx.post = True
    assert asyncio.run(forecast_question(ctx, q)) == "completed"
    assert len(models.calls) == calls  # no new forecasting
    assert api.forecasts == [(q.id, recorded)] and len(api.comments) == 1


def test_a_half_done_question_with_its_forecast_on_the_site_only_gets_its_comment(tmp_path):
    q = binary_q()
    ctx, api, _ = make_ctx(tmp_path, [q], post=False)
    asyncio.run(forecast_question(ctx, q))
    ctx.post = True
    api.on_site.add(q.id)
    q.has_my_forecast = True
    assert asyncio.run(forecast_question(ctx, q)) == "completed"
    assert api.forecasts == [] and len(api.comments) == 1


def test_comments_name_no_model_vendor_or_ai(tmp_path):
    def chatty(qid, member, prompt):
        return ("Summary: As a language model built by OpenAI, GPT thinks rain is likely. "
                "Recent forecasts from the weather service show a storm front. Claude agrees. "
                "AI tools were consulted.\nProbability: 61%")
    q = binary_q()
    ctx, api, _ = make_ctx(tmp_path, [q], replies=chatty)
    asyncio.run(forecast_question(ctx, q))
    text = api.comments[0][1]
    assert "storm front" in text
    assert not UNSAID.search(text)


@pytest.mark.parametrize("make", ALL)
def test_comment_lines_are_clean_for_every_type(tmp_path, make):
    q = make()
    ctx, api, _ = make_ctx(tmp_path, [q])
    asyncio.run(forecast_question(ctx, q))
    assert not UNSAID.search(api.comments[0][1])


def test_one_bad_reply_does_not_stop_the_forecast_and_its_text_stays_out_of_the_log(tmp_path):
    log_file = tmp_path / "bot.log"
    logs.setup(log_file)
    secret_words = "ZEBRA-QUOTE the reply text that must never be logged"
    seen = {"n": 0}

    def flaky(qid, member, prompt):
        if "You are forecasting" not in prompt:
            return "Research brief."
        seen["n"] += 1
        if seen["n"] == 1:
            return secret_words
        return reply_for(binary_q())
    q = binary_q()
    ctx, api, _ = make_ctx(tmp_path, [q], replies=flaky)
    assert asyncio.run(forecast_question(ctx, q)) == "done"
    text = log_file.read_text()
    assert "event=error type=ParseError" in text and "stage=parse" in text
    assert "ZEBRA" not in text and "never be logged" not in text


def test_a_dry_run_logs_no_probability(tmp_path):
    log_file = tmp_path / "bot.log"
    logs.setup(log_file)
    questions = [make(i + 1) for i, make in enumerate(ALL)]
    ctx, api, _ = make_ctx(tmp_path, questions, post=False)
    health = Health(tmp_path)
    for q in questions:
        q.tournament = "fall-test"
    asyncio.run(run_pass(ctx, ["fall-test"], health))
    text = log_file.read_text()
    assert "attempted=6" in text
    for leak in ("43.2", "0.432", "52.5", "40.25", "Rain is common"):
        assert leak not in text
    assert not api.forecasts and not api.comments


def test_reduced_budget_uses_the_light_ensemble(tmp_path):
    q = binary_q()
    ctx, _, models = make_ctx(tmp_path, [q], budget_cap=10.0)
    ctx.budget.spent = 8.5
    asyncio.run(forecast_question(ctx, q))
    used = {m for _, m, _ in models.calls}
    assert used == {m.model for m in pipeline.LIGHT}


def test_minibench_uses_the_light_ensemble_without_web_search(tmp_path):
    q = binary_q(tournament="minibench")
    ctx, _, models = make_ctx(tmp_path, [q])
    asyncio.run(forecast_question(ctx, q))
    assert {m for _, m, _ in models.calls} == {m.model for m in pipeline.LIGHT}
    assert not any(web for _, _, web in models.calls)


def test_main_ensemble_researches_on_the_web_first(tmp_path):
    q = binary_q()
    ctx, _, models = make_ctx(tmp_path, [q])
    asyncio.run(forecast_question(ctx, q))
    assert any(web for _, _, web in models.calls)
    assert {m for _, m, web in models.calls if not web} == {m.model for m in pipeline.MAIN}


def test_spent_budget_stops_the_bot_with_code_3(tmp_path):
    q = binary_q()
    ctx, api, _ = make_ctx(tmp_path, [q], budget_cap=0.05)
    code = asyncio.run(run_forever(ctx, ["fall-test"], Health(tmp_path), max_passes=5, sleep=_no_sleep))
    assert code == EXIT_BUDGET
    assert not api.forecasts


def test_every_question_failing_three_passes_exits_with_code_4(tmp_path):
    q = binary_q()
    ctx, api, _ = make_ctx(tmp_path, [q], replies=lambda *a: "no forecast here")
    code = asyncio.run(run_forever(ctx, ["fall-test"], Health(tmp_path), max_passes=10, sleep=_no_sleep))
    assert code == EXIT_ALL_FAILING


def test_health_file_has_counts_and_spend_only(tmp_path):
    q = binary_q()
    ctx, _, _ = make_ctx(tmp_path, [q])
    health = Health(tmp_path)
    code = asyncio.run(run_forever(ctx, ["fall-test"], health, max_passes=1, sleep=_no_sleep))
    assert code == 0
    import json
    data = json.loads((tmp_path / "health.json").read_text())
    assert data["last_24h"]["done"] == 1 and data["last_24h"]["posted"] == 1
    assert data["spend_usd"] > 0 and data["budget_warning"] is False
    assert "43" not in json.dumps(data["last_24h"])


async def _no_sleep(_seconds):
    return None
