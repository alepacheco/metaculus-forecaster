from datetime import datetime, timezone

import pytest

from forecaster import logs
from forecaster.budget import Budget, BudgetExhausted, QuestionCapReached
from forecaster.runner import next_pass_time


def test_budget_steps_down_then_stops(tmp_path):
    b = Budget(tmp_path / "ledger.json", "s", 10.0, 5.0)
    assert b.mode == "full" and not b.warning
    for qid in range(8):
        b.reserve(qid, 1.0)
        b.settle(qid, 1.0, 1.0)
    assert b.mode == "reduced" and b.warning
    b.reserve(9, 1.0)
    b.settle(9, 1.0, 1.0)
    b.reserve(10, 1.0)
    b.settle(10, 1.0, 1.0)
    assert b.mode == "stopped"
    with pytest.raises(BudgetExhausted):
        b.reserve(11, 0.01)


def test_reservations_stop_concurrent_calls_overshooting(tmp_path):
    b = Budget(None, "s", 1.0, 5.0)
    b.reserve(1, 0.6)
    with pytest.raises(BudgetExhausted):
        b.reserve(2, 0.6)
    b.settle(1, 0.6, 0.1)
    b.reserve(2, 0.6)


def test_per_question_cap(tmp_path):
    b = Budget(None, "s", 100.0, 1.0)
    b.reserve(1, 0.8)
    b.settle(1, 0.8, 0.8)
    with pytest.raises(QuestionCapReached):
        b.reserve(1, 0.3)
    b.reserve(2, 0.3)


def test_ledger_survives_a_restart(tmp_path):
    b = Budget(tmp_path / "ledger.json", "fall", 10.0, 5.0)
    b.reserve(1, 1.0)
    b.settle(1, 1.0, 0.75)
    again = Budget(tmp_path / "ledger.json", "fall", 10.0, 5.0)
    assert again.spent == pytest.approx(0.75) and again.question_spent(1) == pytest.approx(0.75)
    other = Budget(tmp_path / "ledger.json", "spring", 10.0, 5.0)
    assert other.spent == 0.0


def test_event_fields_carry_no_free_text(tmp_path):
    log_file = tmp_path / "bot.log"
    logs.setup(log_file)
    logs.event("listed", tournament="fall-futureeval-2026", note="It will probably rain: 70%")
    text = log_file.read_text()
    assert "tournament=fall-futureeval-2026" in text and "note=<redacted>" in text and "rain" not in text


def test_errors_log_type_and_place_not_message(tmp_path):
    log_file = tmp_path / "bot.log"
    logs.setup(log_file)
    try:
        raise ValueError("the model said 85% yes")
    except ValueError as exc:
        logs.error(exc, stage="parse", question_id=7, question_type="binary")
    text = log_file.read_text()
    assert "type=ValueError" in text and "qid=7" in text and "qtype=binary" in text
    assert "85%" not in text and "model said" not in text


def test_passes_run_off_the_hour():
    at = datetime(2026, 10, 21, 13, 50, 5, tzinfo=timezone.utc)
    assert next_pass_time(at) == datetime(2026, 10, 21, 14, 7, tzinfo=timezone.utc)
    at = datetime(2026, 10, 21, 14, 7, 0, tzinfo=timezone.utc)
    assert next_pass_time(at) == datetime(2026, 10, 21, 14, 27, tzinfo=timezone.utc)
