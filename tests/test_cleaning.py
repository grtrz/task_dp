"""Тесты очистки: по одному на каждый дефект + сквозной прогон на реальных данных.

Запуск: python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dp.cleaning import KEY, CleaningConfig, CleaningLog, build_panel, clean, drop_duplicates, find_unit_breaks
from dp.io import Inputs, load_inputs

DATA = Path(__file__).resolve().parents[1] / "data"
FLAT = [100, 110, 90, 105, 95, 100, 108, 92, 100, 104, 96, 100]


def sales_rows(sku: str, values, start="2023-01", loc="MSK") -> pd.DataFrame:
    return pd.DataFrame({"sku": sku, "location": loc, "period": [str(p) for p in pd.period_range(start, periods=len(values), freq="M")], "qty": values})


def make_inputs(sales: pd.DataFrame, stock=None, promo=None, items=None) -> Inputs:
    if stock is None:
        stock = sales[["sku", "location", "period"]].drop_duplicates().assign(stock_end_qty=100, days_out_of_stock=0)
    if promo is None:
        promo = pd.DataFrame(columns=["sku", "period", "promo_type", "discount_pct"])
    if items is None:
        items = pd.DataFrame({"sku": sorted(sales["sku"].unique()), "category": "X", "abc_class": "A",
                              "launch_period": "2023-01", "predecessor_sku": np.nan, "uom": "PCS"})
    return Inputs(sales=sales, stock=stock, promo=promo, items=items)


def stock_with(sales: pd.DataFrame, oos: dict[tuple[str, str], int]) -> pd.DataFrame:
    st = sales[["sku", "location", "period"]].drop_duplicates().assign(stock_end_qty=100, days_out_of_stock=0)
    for (sku, period), days in oos.items():
        st.loc[(st["sku"] == sku) & (st["period"] == period), "days_out_of_stock"] = days
    return st


def row(panel, sku, period, loc="MSK"):
    return panel[(panel["sku"] == sku) & (panel["location"] == loc) & (panel["period"] == pd.Period(period))].iloc[0]


class DuplicatesTest(unittest.TestCase):
    def test_exact_duplicate_dropped_not_summed(self):
        s = sales_rows("A", FLAT)
        s = pd.concat([s, s.iloc[[3]]], ignore_index=True)
        log = CleaningLog()
        out = drop_duplicates(s, log)
        self.assertEqual(len(out), len(FLAT))
        self.assertEqual(out.loc[out["period"] == "2023-04", "qty"].iloc[0], 105)
        self.assertEqual(log.count("duplicate_exact"), 1)

    def test_conflicting_duplicate_keeps_last_and_is_logged_separately(self):
        s = sales_rows("A", FLAT)
        s = pd.concat([s, s.iloc[[3]].assign(qty=999)], ignore_index=True)
        log = CleaningLog()
        out = drop_duplicates(s, log)
        self.assertEqual(out.loc[out["period"] == "2023-04", "qty"].iloc[0], 999)
        self.assertEqual((log.count("duplicate_exact"), log.count("duplicate_conflict")), (0, 1))


class CalendarTest(unittest.TestCase):
    def test_missing_month_is_gap_not_zero(self):
        panel, _ = clean(make_inputs(sales_rows("A", FLAT).drop(index=5)))
        r = row(panel, "A", "2023-06")
        self.assertEqual(r["flag"], "gap")
        self.assertTrue(np.isnan(r["qty_raw"]))
        self.assertGreater(r["qty"], 90)  # подставлен уровень соседей, а не 0
        self.assertFalse(r["observed"])

    def test_series_calendar_is_not_padded_to_global_end(self):
        s = pd.concat([sales_rows("A", FLAT), sales_rows("B", FLAT[:6])], ignore_index=True)
        panel = build_panel(s, CleaningLog())
        self.assertEqual((panel["sku"] == "B").sum(), 6)

    def test_ended_series_is_dropped_and_logged(self):
        s = pd.concat([sales_rows("A", FLAT), sales_rows("B", FLAT[:6])], ignore_index=True)
        panel, log = clean(make_inputs(s))
        self.assertEqual(set(panel["sku"]), {"A"})
        self.assertEqual(log.count("series_ended"), 1)

    def test_bad_period_format_raises(self):
        s = sales_rows("A", FLAT)
        s.loc[0, "period"] = "01.2023"
        with self.assertRaises(ValueError):
            build_panel(s, CleaningLog())


class NegativesTest(unittest.TestCase):
    def test_negative_is_masked_and_replaced_by_neighbours(self):
        vals = FLAT.copy(); vals[4] = -8
        panel, _ = clean(make_inputs(sales_rows("A", vals)))
        r = row(panel, "A", "2023-05")
        self.assertEqual((r["flag"], r["qty_raw"]), ("negative", -8))
        self.assertGreater(r["qty"], 0)
        self.assertTrue((panel["qty"] >= 0).all())


class StockoutTest(unittest.TestCase):
    def test_zero_in_stockout_month_is_masked_but_ordinary_zero_stays(self):
        vals = FLAT.copy(); vals[3] = 0; vals[8] = 0
        s = sales_rows("A", vals)
        panel, _ = clean(make_inputs(s, stock=stock_with(s, {("A", "2023-04"): 30})))
        oos, honest_zero = row(panel, "A", "2023-04"), row(panel, "A", "2023-09")
        self.assertEqual(oos["flag"], "oos")
        self.assertGreater(oos["qty"], 80)
        self.assertEqual((honest_zero["flag"], honest_zero["qty"]), ("", 0))

    def test_threshold_decides_whether_a_partial_stockout_is_masked(self):
        s = sales_rows("A", FLAT)
        inp = make_inputs(s, stock=stock_with(s, {("A", "2023-04"): 5}))
        self.assertEqual(row(clean(inp)[0], "A", "2023-04")["flag"], "")
        self.assertEqual(row(clean(inp, CleaningConfig(oos_min_days=5))[0], "A", "2023-04")["flag"], "oos")


class PromoTest(unittest.TestCase):
    def test_recurring_promo_spike_does_not_survive_as_seasonality(self):
        vals = FLAT * 2
        vals[10] = 400; vals[22] = 410   # ноябрь обоих лет
        promo = pd.DataFrame({"sku": ["A", "A"], "period": ["2023-11", "2024-11"], "promo_type": "TPR", "discount_pct": 25})
        panel, log = clean(make_inputs(sales_rows("A", vals), promo=promo))
        for p in ("2023-11", "2024-11"):
            r = row(panel, "A", p)
            self.assertEqual(r["flag"], "promo")
            self.assertLess(r["qty"], 150)
        self.assertEqual(log.count("promo"), 2)

    def test_promo_applies_to_its_sku_only(self):
        s = pd.concat([sales_rows("A", FLAT), sales_rows("B", FLAT)], ignore_index=True)
        promo = pd.DataFrame({"sku": ["A"], "period": ["2023-03"], "promo_type": "TPR", "discount_pct": 25})
        panel, _ = clean(make_inputs(s, promo=promo))
        self.assertEqual(row(panel, "B", "2023-03")["flag"], "")


class UnitBreakTest(unittest.TestCase):
    STEP_UP = [200, 210, 190, 205, 195, 200, 200000, 210000, 190000, 205000, 195000, 200000]

    def test_x1000_step_detected(self):
        self.assertEqual(find_unit_breaks(self.STEP_UP), [(6, 1000.0)])

    def test_history_rescaled_to_current_units(self):
        panel, _ = clean(make_inputs(sales_rows("A", self.STEP_UP)))
        self.assertEqual((row(panel, "A", "2023-01")["qty"], row(panel, "A", "2023-01")["qty_raw"]), (200000, 200))
        self.assertEqual(row(panel, "A", "2023-12")["qty"], 200000)

    def test_downward_step_detected(self):
        self.assertEqual(find_unit_breaks([200000, 210000, 190000, 200, 210, 190]), [(3, 0.001)])

    def test_single_huge_outlier_is_not_a_unit_break(self):
        self.assertEqual(find_unit_breaks([100, 110, 90, 100000, 105, 95, 100, 108]), [])

    def test_promo_sized_spike_is_not_a_unit_break(self):
        self.assertEqual(find_unit_breaks([100, 110, 90, 420, 410, 95, 100, 108]), [])

    def test_seasonality_and_intermittency_do_not_trigger(self):
        self.assertEqual(find_unit_breaks([0, 0, 30, 0, 0, 45, 0, 0, 28, 0, 0, 40]), [])
        self.assertEqual(find_unit_breaks([50, 80, 160, 400, 500, 400, 160, 80, 50, 40, 60, 120]), [])

    def test_break_next_to_masked_value_still_found(self):
        v = [200, 210, np.nan, 205, 195, 200, 200000, np.nan, 190000, 205000, 195000, 200000]
        self.assertEqual([t for t, _ in find_unit_breaks(v)], [6])


class StitchTest(unittest.TestCase):
    ITEMS = pd.DataFrame({"sku": ["OLD", "NEW", "LIVE"], "category": "X", "abc_class": "A",
                          "launch_period": ["2023-01", "2023-10", "2023-01"],
                          "predecessor_sku": [np.nan, "OLD", np.nan], "uom": "PCS"})

    def setUp(self):
        old = [500, 520, 480, 510, 490, 500, 300, 290, 310]      # предшественник, закончился
        new = [150, 160, 140, 150, 155, 145]                     # преемник стартует после него
        self.sales = pd.concat([sales_rows("OLD", old), sales_rows("NEW", new, start="2023-10"),
                                sales_rows("LIVE", FLAT + [100] * 3)], ignore_index=True)
        self.panel, self.log = clean(make_inputs(self.sales, items=self.ITEMS))

    def test_history_is_prepended_and_scaled_to_successor_level(self):
        new = self.panel[self.panel["sku"] == "NEW"]
        self.assertEqual((len(new), new["from_predecessor"].sum()), (15, 9))
        self.assertAlmostEqual(row(self.panel, "NEW", "2023-09")["qty"], 310 * 150 / 300)   # масштаб границы 150/300
        self.assertAlmostEqual(row(self.panel, "NEW", "2023-01")["qty"], 500 * 0.5)

    def test_inherited_rows_are_not_observed_and_own_rows_untouched(self):
        self.assertFalse(row(self.panel, "NEW", "2023-05")["observed"])
        self.assertTrue(row(self.panel, "NEW", "2023-10")["observed"])
        self.assertEqual(row(self.panel, "NEW", "2023-10")["qty"], 150)

    def test_dead_predecessor_is_dropped_live_series_stays(self):
        self.assertEqual(set(self.panel["sku"]), {"NEW", "LIVE"})
        self.assertEqual(self.log.count("series_ended"), 1)

    def test_predecessor_masks_carry_over(self):
        promo = pd.DataFrame({"sku": ["OLD"], "period": ["2023-03"], "promo_type": "TPR", "discount_pct": 25})
        panel, _ = clean(make_inputs(self.sales, promo=promo, items=self.ITEMS))
        r = row(panel, "NEW", "2023-03")
        self.assertEqual(r["flag"], "promo")
        self.assertFalse(np.isnan(r["qty"]))


class ImputeTest(unittest.TestCase):
    def test_no_nan_left_and_replaced_rows_flagged(self):
        vals = FLAT.copy(); vals[2] = -1; vals[7] = -1
        panel, _ = clean(make_inputs(sales_rows("A", vals)))
        self.assertFalse(panel["qty"].isna().any())
        self.assertEqual(int((panel["flag"] != "").sum()), 2)

    def test_imputation_uses_median_of_real_neighbours_not_mean(self):
        vals = [100, 100, 100, -1, 100, 100, 5000, 100, 100, 100]
        panel, _ = clean(make_inputs(sales_rows("A", vals)))
        self.assertEqual(row(panel, "A", "2023-04")["qty"], 100)

    def test_imputed_values_do_not_feed_each_other(self):
        vals = [100, 100, 100, -1, -1, -1, 300, 300, 300]
        panel, _ = clean(make_inputs(sales_rows("A", vals)))
        self.assertEqual(row(panel, "A", "2023-05")["qty"], 200)  # окно ±3: 100,100,300,300

    def test_input_frames_are_not_mutated(self):
        s = sales_rows("A", FLAT)
        before = s.copy()
        clean(make_inputs(s))
        pd.testing.assert_frame_equal(s, before)


class RealDataTest(unittest.TestCase):
    """Сквозной прогон: известные дефекты находятся, остальное не трогается."""

    @classmethod
    def setUpClass(cls):
        cls.panel, cls.log = clean(load_inputs(DATA))

    def test_defect_counts(self):
        steps = ("duplicate_exact", "negative", "oos", "promo", "gap", "unit_break", "stitch", "series_ended")
        self.assertEqual({s: self.log.count(s) for s in steps},
                         {"duplicate_exact": 3, "negative": 1, "oos": 10, "promo": 8, "gap": 4,
                          "unit_break": 1, "stitch": 1, "series_ended": 1})

    def test_unit_break_only_on_sku_0018(self):
        hit = self.log.frame().query("step == 'unit_break'").iloc[0]
        self.assertEqual((hit["sku"], hit["period"]), ("SKU_0018", "2024-07"))
        s = self.panel.query("sku == 'SKU_0018'")["qty"]
        self.assertLess(s.max() / s.min(), 4)  # ряд стал одномасштабным

    def test_panel_is_complete_and_sane(self):
        p = self.panel
        self.assertFalse(p["qty"].isna().any())
        self.assertTrue((p["qty"] >= 0).all())
        self.assertFalse(p.duplicated(KEY + ["period"]).any())
        self.assertEqual(p.groupby(KEY).ngroups, 45)                      # 46 исходных рядов − предшественник SKU_0011
        self.assertEqual(p.groupby(KEY)["period"].max().nunique(), 1)     # все ряды доходят до конца истории

    def test_untouched_series_equal_raw(self):
        p = self.panel
        rows = p[(p["flag"] == "") & ~p["from_predecessor"] & (p["sku"] != "SKU_0018")]
        self.assertTrue((rows["qty"] == rows["qty_raw"]).all())

    def test_intermittent_zeros_survive(self):
        self.assertGreater((self.panel.query("sku == 'SKU_0002'")["qty"] == 0).sum(), 20)


if __name__ == "__main__":
    unittest.main()
