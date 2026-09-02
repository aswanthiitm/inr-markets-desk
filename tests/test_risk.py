"""Risk tests: estimator ordering, and backtests against synthetic data with a known answer."""
import numpy as np
import pandas as pd
import pytest

from imd.risk import (
    backtest_var, basel_traffic_light, christoffersen_independence, ewma_volatility,
    historical_var, kupiec_pof, parametric_var, pca_curve_factors, student_t_var,
)

RNG = np.random.default_rng(0)


def test_es_is_always_at_least_var():
    pnl = pd.Series(RNG.standard_normal(5000) * 1e5)
    for estimate in (historical_var(pnl), parametric_var(pnl), student_t_var(pnl)):
        assert estimate.expected_shortfall >= estimate.var


def test_var_rises_with_confidence():
    pnl = pd.Series(RNG.standard_normal(5000) * 1e5)
    assert historical_var(pnl, 0.99).var > historical_var(pnl, 0.95).var


def test_normal_var_matches_the_closed_form():
    pnl = pd.Series(RNG.standard_normal(400_000))
    assert parametric_var(pnl, 0.99).var == pytest.approx(2.3263, abs=0.02)


def test_square_root_of_time_scaling():
    pnl = pd.Series(RNG.standard_normal(20_000))
    one_day = parametric_var(pnl, 0.99, 1).var
    ten_day = parametric_var(pnl, 0.99, 10).var
    assert ten_day == pytest.approx(one_day * np.sqrt(10), rel=1e-9)


def test_student_t_finds_fat_tails_the_normal_model_misses():
    fat = pd.Series(RNG.standard_t(3, 20_000))
    assert student_t_var(fat, 0.99).var > parametric_var(fat, 0.99).var


def test_kupiec_accepts_a_correctly_calibrated_model():
    exceptions = RNG.random(2500) < 0.01
    assert kupiec_pof(exceptions, 0.99)["p_value"] > 0.05


def test_kupiec_rejects_a_model_that_breaks_far_too_often():
    exceptions = RNG.random(2500) < 0.06
    assert kupiec_pof(exceptions, 0.99)["reject_5pct"]


def test_christoffersen_rejects_clustered_exceptions():
    exceptions = np.zeros(1000, dtype=int)
    exceptions[300:340] = 1  # every breach in one run
    assert christoffersen_independence(exceptions)["reject_5pct"]


def test_basel_zones():
    assert basel_traffic_light(4) == "GREEN"
    assert basel_traffic_light(7) == "AMBER"
    assert basel_traffic_light(11) == "RED"


def test_ewma_volatility_reacts_to_a_regime_break():
    calm = RNG.standard_normal(500) * 0.002
    stormy = RNG.standard_normal(200) * 0.02
    series = pd.Series(np.concatenate([calm, stormy]),
                       index=pd.date_range("2024-01-01", periods=700, freq="B"))
    vol = ewma_volatility(series)
    assert vol.iloc[-1] > 4 * vol.iloc[480]


def test_pca_recovers_a_level_factor_from_a_parallel_shifting_curve():
    dates = pd.date_range("2024-01-01", periods=400, freq="B")
    shifts = np.cumsum(RNG.standard_normal(400)) * 0.05
    panel = pd.DataFrame({t: 6.0 + shifts for t in [1, 2, 5, 10, 30]}, index=dates)
    result = pca_curve_factors(panel)
    assert result["explained_variance_ratio"]["level"] > 0.99
    assert np.allclose(result["loadings"]["level"], result["loadings"]["level"].iloc[0], atol=1e-8)


def test_backtest_reports_every_exception_it_should():
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    pnl = pd.Series(np.full(300, -1.0), index=idx)
    pnl.iloc[:5] = -100.0
    var = pd.Series(np.full(300, 10.0), index=idx)
    assert backtest_var(pnl, var)["exceptions"] == 5
