"""Hedging simulator tests."""
import numpy as np
import pandas as pd

from imd.hedging import ExposureSchedule, HedgePolicy, compare_policies, simulate_paths

SPOT = 95.0


def _setup():
    exposure = ExposureSchedule.monthly(12, 1e6)
    forwards = SPOT * (1 + 0.015 * exposure.tenors_years)
    returns = pd.Series(np.random.default_rng(3).standard_normal(2000) * 0.004)
    paths = simulate_paths(SPOT, exposure.tenors_years, 4000, returns=returns, seed=1)
    return exposure, forwards, paths


def test_a_full_forward_hedge_removes_all_uncertainty():
    exposure, forwards, paths = _setup()
    table = compare_policies(
        [HedgePolicy("unhedged", "unhedged"), HedgePolicy("forward", "forward", 1.0)],
        exposure, SPOT, forwards, 0.0575, 0.0415, paths,
    )
    assert table.loc[table.policy == "forward", "std_cost_cr"].item() < 1e-9
    assert table.loc[table.policy == "forward", "vol_reduction_pct"].item() > 99.999


def test_a_collar_caps_the_worst_case_but_not_to_a_point():
    exposure, forwards, paths = _setup()
    table = compare_policies(
        [HedgePolicy("unhedged", "unhedged"),
         HedgePolicy("collar", "collar", 1.0, cap_over_forward=0.02, vol=0.05)],
        exposure, SPOT, forwards, 0.0575, 0.0415, paths,
    )
    unhedged = table.loc[table.policy == "unhedged"].iloc[0]
    collar = table.loc[table.policy == "collar"].iloc[0]
    assert collar["worst_1pct_rate"] < unhedged["worst_1pct_rate"]
    assert 0.0 < collar["std_cost_cr"] < unhedged["std_cost_cr"]


def test_partial_hedging_sits_between_the_two_extremes():
    exposure, forwards, paths = _setup()
    table = compare_policies(
        [HedgePolicy("unhedged", "unhedged"),
         HedgePolicy("half", "forward", 0.5),
         HedgePolicy("full", "forward", 1.0)],
        exposure, SPOT, forwards, 0.0575, 0.0415, paths,
    ).set_index("policy")
    assert table.loc["full", "std_cost_cr"] < table.loc["half", "std_cost_cr"] < table.loc["unhedged", "std_cost_cr"]


def test_bootstrap_paths_keep_the_shape_of_the_input_returns():
    returns = pd.Series(np.random.default_rng(5).standard_normal(3000) * 0.005)
    paths = simulate_paths(SPOT, np.array([1.0]), 20_000, returns=returns, seed=2)
    realised = np.std(np.log(paths[:, 0] / SPOT))
    assert realised == np.float64(realised)
    assert 0.5 < realised / (0.005 * np.sqrt(252)) < 1.6
