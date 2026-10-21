import numpy as np
import pytest

from fakes import date_q, discrete_q, log_q, numeric_q, reply_for
from forecaster import aggregate, cdf
from forecaster.parse import parse_percentiles


@pytest.mark.parametrize("make", [numeric_q, log_q, discrete_q, date_q])
def test_cdf_from_percentiles_is_valid(make):
    q = make()
    points = parse_percentiles(reply_for(q), q)
    raw = cdf.percentiles_to_cdf(points, q)
    final = cdf.standardize(raw, q)
    assert len(final) == q.cdf_size
    assert cdf.problems(final, q) == []


@pytest.mark.parametrize("make", [numeric_q, log_q, discrete_q, date_q])
def test_aggregate_of_members_is_valid(make):
    q = make()
    points = parse_percentiles(reply_for(q), q)
    shifted = [(p, v * 1.1 if q.type != "date" else v + 86400 * 20) for p, v in points]
    members = [cdf.percentiles_to_cdf(points, q), cdf.percentiles_to_cdf(shifted, q)]
    final = aggregate.continuous(members, q)
    assert cdf.problems(final, q) == []


def test_closed_bounds_hold_no_outside_mass():
    q = log_q()
    final = cdf.standardize(cdf.percentiles_to_cdf(parse_percentiles(reply_for(q), q), q), q)
    assert final[0] == 0.0 and final[-1] == 1.0


def test_open_bounds_keep_minimum_outside_mass():
    q = numeric_q()
    points = [(0.1, 40.0), (0.5, 50.0), (0.9, 60.0)]
    final = cdf.standardize(cdf.percentiles_to_cdf(points, q), q)
    assert final[0] >= 0.001 and final[-1] <= 0.999


def test_a_very_narrow_forecast_is_capped_per_bin():
    q = numeric_q()
    points = [(0.01, 49.9), (0.5, 50.0), (0.99, 50.1)]
    final = cdf.standardize(cdf.percentiles_to_cdf(points, q), q)
    assert cdf.problems(final, q) == []
    assert max(np.diff(final)) <= cdf.bin_cap(q)


def test_median_survives_the_round_trip():
    q = numeric_q()
    points = parse_percentiles(reply_for(q), q)
    final = cdf.standardize(cdf.percentiles_to_cdf(points, q), q)
    assert abs(cdf.quantile(final, q, 0.5) - 43.21) < 2.0


def test_all_mass_outside_a_closed_range_is_refused():
    q = log_q()
    with pytest.raises(ValueError):
        cdf.percentiles_to_cdf([(0.1, 20000.0), (0.5, 30000.0), (0.9, 40000.0)], q)


def test_location_round_trip_on_log_scale():
    q = log_q()
    for value in (1.0, 10.0, 500.0, 10000.0):
        assert abs(cdf.location_to_value(cdf.value_to_location(value, q), q) - value) < 1e-6 * value


def test_a_member_wholly_outside_a_range_with_one_closed_bound_is_refused():
    upper_closed = numeric_q(open_lower_bound=True, open_upper_bound=False)
    with pytest.raises(ValueError):
        cdf.percentiles_to_cdf([(0.1, 200.0), (0.5, 300.0), (0.9, 400.0)], upper_closed)
    lower_closed = numeric_q(open_lower_bound=False, open_upper_bound=True)
    with pytest.raises(ValueError):
        cdf.percentiles_to_cdf([(0.1, -400.0), (0.5, -300.0), (0.9, -200.0)], lower_closed)


def test_minimum_increment_holds_exactly_on_a_flat_forecast():
    from fakes import Question
    q = Question(id=9, post_id=1009, type="discrete", title="x", tournament="t", range_min=0.095, range_max=0.805,
                 open_lower_bound=True, open_upper_bound=True, inbound_outcome_count=71)
    final = cdf.standardize(cdf.percentiles_to_cdf([(0.1, 0.2), (0.5, 0.3), (0.9, 0.4)], q), q)
    assert np.diff(final).min() >= 0.01 / 71
    assert cdf.problems(final, q) == []
