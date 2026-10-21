import pytest

from fakes import choice_q, date_q, numeric_q
from forecaster import aggregate
from forecaster.parse import ParseError, parse_binary, parse_multiple_choice, parse_percentiles, summary


def test_binary_takes_the_last_probability():
    assert parse_binary("Base rate Probability: 20%\n...\n**Probability: 37%**") == pytest.approx(0.37)


def test_binary_without_a_probability_fails():
    with pytest.raises(ParseError):
        parse_binary("I think it is likely.")


def test_multiple_choice_reads_each_option_and_normalises():
    q = choice_q()
    probs = parse_multiple_choice("- Red Party: 50%\n**Blue Party**: 30%\nOther: 30%", q.options)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] == pytest.approx(50 / 110)


def test_multiple_choice_missing_an_option_fails():
    with pytest.raises(ParseError):
        parse_multiple_choice("Red Party: 50%\nBlue Party: 50%", choice_q().options)


def test_percentiles_with_separators_and_disorder():
    q = numeric_q(range_max=100000.0)
    text = "\n".join(f"Percentile {p}: {v}" for p, v in
                     [(10, "1,000"), (50, "5,000"), (90, "4,000"), (20, "2,000"), (80, "9,000")])
    points = parse_percentiles(text, q)
    values = [v for _, v in points]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert [p for p, _ in points] == [0.1, 0.2, 0.5, 0.8, 0.9]


def test_percentiles_for_dates():
    q = date_q()
    text = "\n".join(f"Percentile {p}: 2026-0{i + 1}-15" for i, p in enumerate((5, 20, 50, 80, 95)))
    points = parse_percentiles(text, q)
    assert points[0][1] < points[-1][1]


def test_too_few_percentiles_fail():
    with pytest.raises(ParseError):
        parse_percentiles("Percentile 50: 10", numeric_q())


def test_summary_stops_before_the_final_answer():
    text = "Thinking...\nSummary: Polls favour Red. Turnout is uncertain.\nRed Party: 60%\nBlue Party: 30%\nOther: 10%"
    assert summary(text) == "Polls favour Red. Turnout is uncertain."


def test_binary_aggregate_is_a_capped_median():
    assert aggregate.binary([0.3, 0.4, 0.9]) == pytest.approx(0.4)
    assert aggregate.binary([0.99, 0.995, 0.97]) == aggregate.BINARY_MAX
    assert aggregate.binary([0.001, 0.002, 0.01]) == aggregate.BINARY_MIN


def test_choice_aggregate_has_a_floor_and_sums_to_one():
    probs = aggregate.multiple_choice([[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.97, 0.03, 0.0]])
    assert sum(probs) == pytest.approx(1.0)
    assert min(probs) >= aggregate.CHOICE_FLOOR - 1e-9
