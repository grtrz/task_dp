"""Тесты сегментации, методов, выбора и валидации. Запуск: python -m unittest discover -s tests -t ."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import dp.selection as sel
from dp.methods import REGISTRY, CrostonSBA, SeasonalNaive, SeasonalSES, predict, seasonal_indices
from dp.pipeline import run
from dp.segmentation import classify
from dp.selection import CANDIDATES, Rule, select_method
from dp.validation import _one_fold, backtest

DATA = Path(__file__).resolve().parents[1] / "data"
PATTERN = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.3, 1.1, 1.0, 0.9, 0.9, 1.0])


def series(values, start="2023-01") -> pd.Series:
    return pd.Series(np.asarray(values, float), index=pd.period_range(start, periods=len(values), freq="M"))


def seasonal_series(n=36, level=100.0, noise=0.0, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return series(level * PATTERN[np.arange(n) % 12] * (1 + noise * rng.standard_normal(n)))


class SeasonalIndicesTest(unittest.TestCase):
    def test_indices_recovered(self):
        np.testing.assert_allclose(seasonal_indices(seasonal_series()), PATTERN / PATTERN.mean(), atol=1e-6)

    def test_short_series_gets_flat_indices(self):
        np.testing.assert_array_equal(seasonal_indices(seasonal_series(n=20)), np.ones(12))

    def test_month_of_zeros_does_not_produce_zero_index_or_infinite_forecast(self):
        v = PATTERN[np.arange(36) % 12] * 100
        v[np.arange(36) % 12 == 4] = 0          # каждый май — нули (например, немаскированный дефицит)
        y = series(v)
        self.assertTrue((seasonal_indices(y) > 0).all())
        self.assertTrue(np.isfinite(SeasonalSES().forecast(y, 3)).all())


class SegmentationTest(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(classify(series([100] * 5)), "short")
        self.assertEqual(classify(series([0, 0, 30] * 12)), "intermittent")
        self.assertEqual(classify(series([0] * 36)), "intermittent")
        self.assertEqual(classify(seasonal_series(noise=0.03)), "regular")

    def test_short_wins_over_intermittent(self):
        self.assertEqual(classify(series([0, 0, 5, 0, 0, 7])), "short")


class MethodsTest(unittest.TestCase):
    def test_snaive_repeats_last_year(self):
        y = seasonal_series()
        np.testing.assert_allclose(SeasonalNaive().forecast(y, 3), y.to_numpy()[-12:-9])

    def test_snaive_falls_back_to_naive_on_short_history(self):
        np.testing.assert_array_equal(SeasonalNaive().forecast(series([1, 2, 3]), 2), [3, 3])

    def test_ses_seasonal_follows_pattern_on_clean_data(self):
        np.testing.assert_allclose(SeasonalSES().forecast(seasonal_series(), 3), 100 * PATTERN[:3], rtol=0.02)

    def test_croston_on_regular_intermittent_demand(self):
        self.assertAlmostEqual(CrostonSBA().forecast(series([0, 10] * 12), 1)[0], 0.95 * 10 / 2)   # (1 - alpha/2) * размер / интервал

    def test_croston_all_zero(self):
        np.testing.assert_array_equal(CrostonSBA().forecast(series([0] * 12), 3), [0, 0, 0])

    def test_every_method_returns_h_finite_values(self):
        y = seasonal_series(noise=0.05)
        for name, m in REGISTRY.items():
            f = m.forecast(y, 3)
            self.assertEqual(len(f), 3, name)
            self.assertTrue(np.isfinite(f).all(), name)


class ValidationTest(unittest.TestCase):
    def test_no_leakage_forecast_ignores_values_after_origin(self):
        y = seasonal_series(noise=0.05)
        obs = pd.Series(True, index=y.index)
        origin = y.index[-4]
        y2 = y.copy(); y2[y.index > origin] *= 5
        for m in ("snaive", "ses_seasonal", "croston_sba"):
            a, b = _one_fold(y, obs, origin, 3, m), _one_fold(y2, obs, origin, 3, m)
            np.testing.assert_array_equal(a["forecast"], b["forecast"])

    def test_only_observed_months_are_scored(self):
        y = seasonal_series()
        obs = pd.Series(True, index=y.index)
        origin = y.index[-4]
        obs[origin + 2] = False
        self.assertEqual(list(_one_fold(y, obs, origin, 3, "snaive")["period"]), [origin + 1, origin + 3])

    def test_forecast_is_non_negative(self):
        class Negative:
            name = "negative"

            def forecast(self, y, h):
                return np.full(h, -5.0)

        y = seasonal_series()
        with mock.patch.dict(REGISTRY, {"negative": Negative()}):
            self.assertTrue((predict("negative", y, 3) == 0).all())
            self.assertTrue((_one_fold(y, pd.Series(True, index=y.index), y.index[-4], 3, "negative")["forecast"] == 0).all())

    def test_window_truncates_training_like_a_new_sku(self):
        y = seasonal_series(noise=0.05)
        key, origin = ("A", "MSK"), y.index[-4]
        bt = backtest({key: (y, pd.Series(True, index=y.index))}, {key: "short"}, CANDIDATES, [origin], 3, min_train=5, window=5)
        self.assertEqual(set(bt["segment"]), {"short"})
        self.assertEqual(set(bt["method"]), set(CANDIDATES["short"]))
        self.assertAlmostEqual(bt["mu"].iloc[0], y[:origin].iloc[-5:].mean())


class SelectionTest(unittest.TestCase):
    @staticmethod
    def scores(rows):
        return pd.DataFrame([dict(segment="s", method=m, wape=w, bias=b, rmsse=w, n_series=n) for m, w, b, n in rows])

    def setUp(self):
        sel.BASELINE["s"], sel.PRIMARY["s"] = "base", "wape"
        self.addCleanup(lambda: (sel.BASELINE.pop("s"), sel.PRIMARY.pop("s")))

    def test_must_beat_baseline_by_margin(self):
        m, _ = select_method(self.scores([("base", 0.20, 0.0, 10), ("fancy", 0.199, 0.0, 10)]), "s")
        self.assertEqual(m, "base")
        m, _ = select_method(self.scores([("base", 0.20, 0.0, 10), ("fancy", 0.15, 0.0, 10)]), "s")
        self.assertEqual(m, "fancy")

    def test_biased_winner_is_rejected(self):
        m, _ = select_method(self.scores([("base", 0.20, 0.0, 10), ("fancy", 0.10, -0.25, 10)]), "s")
        self.assertEqual(m, "base")

    def test_biased_baseline_still_eligible(self):
        m, _ = select_method(self.scores([("base", 0.20, 0.30, 10), ("fancy", 0.10, 0.50, 10)]), "s")
        self.assertEqual(m, "base")

    def test_too_few_series_keeps_baseline(self):
        m, why = select_method(self.scores([("base", 0.20, 0.0, 2), ("fancy", 0.05, 0.0, 2)]), "s", Rule(min_series=3))
        self.assertEqual(m, "base")
        self.assertIn("мало рядов", why)

    def test_undefined_baseline_metric_keeps_baseline(self):
        m, _ = select_method(self.scores([("base", np.nan, 0.0, 10), ("fancy", 0.10, 0.0, 10)]), "s")
        self.assertEqual(m, "base")


class PipelineTest(unittest.TestCase):
    def test_end_to_end_forecast_file(self):
        with tempfile.TemporaryDirectory() as out:
            _, summary, _ = run(DATA, out)
            f = pd.read_csv(Path(out) / "forecast.csv")
        self.assertEqual(len(f), 45 * 3)
        self.assertEqual(sorted(f["period"].unique()), ["2026-01", "2026-02", "2026-03"])
        self.assertTrue(np.isfinite(f["forecast"]).all() and (f["forecast"] >= 0).all())
        self.assertFalse(f.duplicated(["sku", "location", "period"]).any())
        self.assertNotIn("SKU_0011", set(f["sku"]))        # предшественник влит в SKU_0036
        self.assertEqual(set(summary["segment"]), {"regular", "intermittent", "short"})

    def test_series_that_ended_is_not_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            for name in ("sales_history", "stock_history", "promo_calendar", "item_master"):
                df = pd.read_csv(DATA / f"{name}.csv")
                if "period" in df and "sku" in df:
                    df = df[~((df["sku"] == "SKU_0010") & (df["period"] > "2025-06"))]
                df.to_csv(data / f"{name}.csv", index=False)
            forecast, _, _ = run(data, Path(tmp) / "out")
        self.assertNotIn("SKU_0010", set(forecast["sku"]))
        self.assertEqual(len(forecast), 44 * 3)


if __name__ == "__main__":
    unittest.main()
