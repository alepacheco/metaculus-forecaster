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


def test_percentiles_with_separators_listed_out_of_order():
    q = numeric_q(range_max=100000.0)
    text = "\n".join(f"Percentile {p}: {v}" for p, v in
                     [(10, "1,000"), (50, "5,000"), (90, "12,000"), (20, "2,000"), (80, "9,000")])
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


def test_multiple_choice_list_formats():
    q = choice_q()
    for text in ("1. Red Party: 50%\n2. Blue Party: 30%\n3. Other: 20%",
                 "A) Red Party: 50%\nB) Blue Party: 30%\nC) Other: 20%",
                 '"Red Party": 50%\n"Blue Party": 30%\n"Other": 20%',
                 "Red Party - 50%\nBlue Party - 30%\nOther - 20%",
                 "Red Party = 50%\nBlue Party = 30%\nOther = 20%"):
        assert parse_multiple_choice(text, q.options) == pytest.approx([0.5, 0.3, 0.2]), text


def test_multiple_choice_labels_that_prefix_each_other():
    probs = parse_multiple_choice("Yes: 60%\nYes, partially: 30%\nNo: 10%", ["Yes", "Yes, partially", "No"])
    assert probs == pytest.approx([0.6, 0.3, 0.1])


def test_numbers_with_magnitude_words_are_refused():
    q = numeric_q(range_min=4e6, range_max=1e7, nominal_min=4e6, nominal_max=1e7)
    for line in ("7.2 trillion", "5k", "2.5M", "3 x 10^6", "1.1 million"):
        text = "\n".join(f"Percentile {p}: {line}" for p in (10, 20, 50, 80, 90))
        with pytest.raises(ParseError):
            parse_percentiles(text, q)


def test_plain_and_scientific_numbers_are_read():
    q = numeric_q(range_min=0.0, range_max=1e7)
    text = "\n".join(f"Percentile {p}: {v}" for p, v in
                     [(10, "$1,200"), (20, "2e3"), (50, "4500.5"), (80, "1.2e6"), (90, "2,000,000 (in units)")])
    assert [v for _, v in parse_percentiles(text, q)] == [1200.0, 2000.0, 4500.5, 1.2e6, 2e6]


def test_percentile_values_that_go_down_are_refused():
    q = numeric_q()
    text = "\n".join(f"Percentile {p}: {v}" for p, v in [(10, 50), (20, 40), (50, 30), (80, 20), (90, 10)])
    with pytest.raises(ParseError):
        parse_percentiles(text, q)


def test_ties_on_a_log_scale_stay_where_the_model_put_them():
    from fakes import log_q
    q = log_q()  # 1 .. 10000, log scaled
    text = "\n".join(f"Percentile {p}: {v}" for p, v in
                     [(1, 1), (5, 1), (10, 1), (40, 100), (50, 100), (60, 100), (90, 5000), (99, 9000)])
    points = parse_percentiles(text, q)
    by_level = dict(points)
    assert by_level[0.5] == pytest.approx(100, rel=1e-3)
    assert by_level[0.1] == pytest.approx(1, rel=1e-3)
    assert [v for _, v in points] == sorted(v for _, v in points)


def test_other_percentile_spellings():
    q = numeric_q()
    text = "P10: 10\nP20: 20\n50th percentile: 30\nP80: 40\n90th percentile: 50"
    assert [v for _, v in parse_percentiles(text, q)] == [10, 20, 30, 40, 50]


def test_binary_variants():
    assert parse_binary("Probability (Yes): ~35%") == pytest.approx(0.35)
