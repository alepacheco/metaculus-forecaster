import asyncio
import json
import re
import time

import httpx
import pytest

from fakes import FakeApi, FakeModels, binary_q, numeric_q
from forecaster import logs, pipeline
from forecaster.__main__ import main
from forecaster.budget import Budget
from forecaster.health import Health
from forecaster.llm import EMPTY_REPLY, Member, ModelClient, ModelError
from forecaster.pipeline import Context, forecast_question
from forecaster.records import Records
from forecaster.roster import NOT_FOUND_SECONDS, Roster
from forecaster.runner import EXIT_ALL_FAILING, run_forever

INKLING = "thinkingmachines/inkling:free"
NEMOTRON = "nvidia/nemotron-3-ultra-550b-a55b:free"
FREE_IDS = {m.model for m in pipeline.FREE + pipeline.FREE_RESERVE}


def free_ctx(tmp_path, questions, **models_kw):
    budget = Budget(tmp_path / "ledger.json", "test", 0.0, 1.0)
    models = FakeModels(budget, cost=0.0, **models_kw)
    models.questions = {q.id: q for q in questions}
    api = FakeApi(questions)
    ctx = Context(api=api, models=models, news=None, budget=budget, records=Records(tmp_path / "records"))
    return ctx, api, models


def test_a_withdrawn_free_model_is_replaced_within_the_same_question(tmp_path):
    first, second = binary_q(1), binary_q(2)
    ctx, api, models = free_ctx(tmp_path, [first, second], gone_models={INKLING})
    assert asyncio.run(forecast_question(ctx, first)) == "done"
    record = ctx.records.load(first.id)
    assert len([m for m in record["members"] if "forecast" in m]) == len(pipeline.FREE)
    used = [m["model"] for m in record["members"]]
    reserve = [m.model for m in pipeline.FREE_RESERVE]
    assert INKLING not in used and reserve[0] in used and reserve[1] in used
    assert 1 <= sum(1 for c in models.calls if c[1] == INKLING) <= 2  # at most once for each of its slots
    models.calls.clear()
    assert asyncio.run(forecast_question(ctx, second)) == "done"
    assert INKLING not in {c[1] for c in models.calls}  # not asked again for a day
    assert len(models.calls) == len(pipeline.FREE) and len(api.forecasts) == 2


def test_a_model_the_provider_no_longer_lists_is_never_called(tmp_path):
    q = binary_q()
    ctx, api, models = free_ctx(tmp_path, [q], listed=FREE_IDS - {NEMOTRON})
    assert asyncio.run(forecast_question(ctx, q)) == "done"
    called = [c[1] for c in models.calls]
    assert NEMOTRON not in called and len(called) == len(pipeline.FREE)
    assert pipeline.models_out(ctx) == [NEMOTRON]


def test_an_unreadable_model_list_leaves_the_ensemble_as_written(tmp_path):
    q = binary_q()
    ctx, _, models = free_ctx(tmp_path, [q], listed=None)
    asyncio.run(forecast_question(ctx, q))
    assert sorted(c[1] for c in models.calls) == sorted(m.model for m in pipeline.FREE)


def test_health_names_the_models_out(tmp_path):
    q = binary_q()
    ctx, _, _ = free_ctx(tmp_path, [q], gone_models={INKLING})
    code = asyncio.run(run_forever(ctx, ["fall-test"], Health(tmp_path), max_passes=1, sleep=_no_sleep))
    assert code == 0
    data = json.loads((tmp_path / "health.json").read_text())
    assert data["models_out"] == [INKLING]
    assert data["errors_24h"].get("member1:ModelError:404", 0) >= 1
    assert data["last_24h"]["posted"] == 1


def test_every_free_model_gone_still_fails_loudly(tmp_path):
    questions = [binary_q(1), binary_q(2)]
    ctx, api, models = free_ctx(tmp_path, questions, gone_models=FREE_IDS)
    code = asyncio.run(run_forever(ctx, ["fall-test"], Health(tmp_path), max_passes=10, sleep=_no_sleep))
    assert code == EXIT_ALL_FAILING and not api.forecasts
    first_question_calls = [c for c in models.calls if c[0] == 1]
    assert len(first_question_calls) <= pipeline.MAX_ATTEMPTS * len(pipeline.FREE) * pipeline.MAX_TRIES_PER_MEMBER


def test_the_roster_fills_from_the_members_still_in_when_the_reserve_is_spent():
    roster = Roster()
    a, b, c = Member("lab/a:free"), Member("lab/b:free"), Member("lab/c:free")
    spare = Member("lab/spare:free")
    roster.listed = {"lab/a:free", "lab/spare:free"}
    assert [m.model for m in roster.ensemble([a, b, c, a, b], [spare])] == \
        ["lab/a:free", "lab/spare:free", "lab/a:free", "lab/a:free", "lab/spare:free"]
    roster.listed = {"lab/z:free"}
    assert roster.ensemble([a, b], [spare]) == [a, b]  # nothing known to be in: as written


def test_a_model_that_answered_not_found_is_tried_again_after_a_day():
    roster = Roster()
    roster.mark_not_found("lab/a:free")
    assert not roster.is_in("lab/a:free")
    roster.not_found["lab/a:free"] = time.time() - NOT_FOUND_SECONDS - 1
    assert roster.is_in("lab/a:free")


def test_the_roster_reads_the_provider_list_at_most_hourly(tmp_path):
    class Lister:
        reads = 0

        async def listed_models(self):
            Lister.reads += 1
            return {"lab/a:free"}
    roster = Roster()

    async def twice():
        await roster.refresh(Lister())
        await roster.refresh(Lister())
    asyncio.run(twice())
    assert Lister.reads == 1 and roster.listed == {"lab/a:free"}


def _client_answering(contents, finish="length"):
    replies = list(contents)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        text = replies.pop(0)
        return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": finish}],
                                          "usage": {"cost": 0}})
    client = ModelClient("key", Budget(None, "t", 0.0, 1.0))
    client._client = httpx.AsyncClient(base_url="https://example.invalid", transport=httpx.MockTransport(handler))
    return client, seen


def test_an_empty_reply_is_asked_for_once_more():
    client, seen = _client_answering(["", "Probability: 40%"])
    reply = asyncio.run(client.complete(1, Member("lab/a:free", estimate_usd=0.0), "prompt"))
    assert reply.text == "Probability: 40%" and len(seen) == 2


def test_two_empty_replies_fail_with_the_finish_reason_logged_not_any_text(tmp_path):
    log_file = tmp_path / "bot.log"
    logs.setup(log_file)
    client, seen = _client_answering(["  ", ""], finish="length")
    with pytest.raises(ModelError) as raised:
        asyncio.run(client.complete(1, Member("lab/a:free", estimate_usd=0.0), "prompt"))
    assert raised.value.status == EMPTY_REPLY and raised.value.reason == "length" and len(seen) == 2
    logs.error(raised.value, stage="member3", question_id=1, question_type="binary", model="lab/a:free")
    text = log_file.read_text()
    assert "stage=member3" in text and "model=lab/a:free" in text and "status=-1 reason=length" in text


def test_the_provider_list_drops_models_past_their_expiry():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "lab/a:free", "expiration_date": "2000-01-01"},
                                                   {"id": "lab/b:free", "expiration_date": None},
                                                   {"id": "lab/c:free", "expiration_date": "2999-12-31"}]})
    client = ModelClient("key", Budget(None, "t", 0.0, 1.0))
    client._client = httpx.AsyncClient(base_url="https://example.invalid", transport=httpx.MockTransport(handler))
    assert asyncio.run(client.listed_models()) == {"lab/b:free", "lab/c:free"}


def test_any_logged_error_sets_the_last_error_type(tmp_path):
    q = binary_q()
    ctx, _, _ = free_ctx(tmp_path, [q], fail_models={NEMOTRON})
    logs._LAST.clear()
    asyncio.run(run_forever(ctx, ["fall-test"], Health(tmp_path), max_passes=1, sleep=_no_sleep))
    data = json.loads((tmp_path / "health.json").read_text())
    assert data["last_error_type"] == "ModelError" and data["last_error_at"]


N = r"[\d.,]+"


@pytest.mark.parametrize("unit,line", [
    ("$", rf"Forecast: median \${N}; 80% interval \${N} to \${N}\."),
    ("$ billion", rf"Forecast: median \${N} billion; 80% interval \${N} to \${N} billion\."),
    ("units", rf"Forecast: median {N} units; 80% interval {N} to {N} units\."),
])
def test_a_currency_symbol_goes_before_the_number(tmp_path, unit, line):
    q = numeric_q(unit=unit)
    ctx, api, _ = free_ctx(tmp_path, [q])
    asyncio.run(forecast_question(ctx, q))
    assert re.fullmatch(line, api.comments[0][1].splitlines()[0])


def test_a_pass_takes_only_the_configs_tournaments_or_the_testing_area(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('tournaments = ["fall-test"]\nseason_budget_usd = 0\n')
    (tmp_path / "bot.env").write_text("METACULUS_TOKEN=x\n")
    monkeypatch.setenv("FORECASTER_LIVE_DIR", str(tmp_path))
    monkeypatch.setenv("FORECASTER_SECRET_FILE", str(tmp_path / "bot.env"))
    assert main(["pass", "--tournament", "some-other-tournament"]) == 1
    assert main(["pass", "--tournament", "fall-test", "--tournament", "elsewhere"]) == 1
    assert not (tmp_path / "health.json").exists()


async def _no_sleep(_seconds):
    return None
