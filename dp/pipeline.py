"""Сборка: очистка -> сегментация -> бэктест -> выбор метода по сегменту -> прогноз."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import clean
from .io import load_inputs
from .methods import predict
from .segmentation import segment_series
from .selection import BASELINE, CANDIDATES, PRIMARY, select_method
from .validation import backtest, build_series, segment_scores

HORIZON = 3


def run(data_dir: str | Path = "data", out_dir: str | Path = "output") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Пишет CSV в out_dir; возвращает (прогноз, таблица сегментов, журнал очистки)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel, log = clean(load_inputs(data_dir))
    segs = segment_series(panel)
    seg_of = segs["segment"].to_dict()
    series = build_series(panel)
    end = panel["period"].max()

    bt = backtest(series, seg_of, CANDIDATES, [end - 9, end - 6, end - 3], HORIZON)
    short = segs[segs["segment"] == "short"]
    if not short.empty:
        # У настоящих коротких рядов (старт 2025-08) своего бэктеста нет: на 5 точках не прогнать ни
        # одного фолда. Имитируем: зрелые ряды обрезаем до такого же окна на том же календарном месте
        # (авг–дек перед горизонтом янв–мар) и проверяем, какой метод справляется с недостатком истории.
        n = int(short["n_months"].median())
        mimic = {k: "short" for k, s in seg_of.items() if s == "regular"}
        bt = pd.concat([bt, backtest(series, mimic, CANDIDATES, [end - 12], HORIZON, min_train=n, window=n)], ignore_index=True)
    scores = segment_scores(bt)
    chosen = {seg: select_method(scores, seg) for seg in segs["segment"].unique()}

    future = pd.period_range(end + 1, periods=HORIZON, freq="M")
    rows = []
    for key, (y, _) in series.items():
        seg = seg_of[key]
        method = chosen[seg][0]
        f = predict(method, y, HORIZON)
        rows += [{"sku": key[0], "location": key[1], "period": str(p), "forecast": round(float(v), 1),
                  "segment": seg, "method": method} for p, v in zip(future, f)]
    forecast = pd.DataFrame(rows)
    if not np.isfinite(forecast["forecast"]).all():
        raise ValueError("в прогнозе есть нечисловые значения:\n" + forecast[~np.isfinite(forecast["forecast"])].to_string())

    summary = _segment_table(segs["segment"].value_counts(), scores, chosen)
    log = log.frame()

    forecast.to_csv(out / "forecast.csv", index=False)
    for name, df in (("backtest", bt), ("scores", scores), ("segment_report", summary)):
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.10g")   # шум в 16-м знаке не должен менять файлы между запусками
    panel.to_csv(out / "clean_history.csv", index=False)
    log.to_csv(out / "cleaning_log.csv", index=False)
    return forecast, summary, log


def _segment_table(n_series: pd.Series, scores: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    s = scores.set_index(["segment", "method"])
    rows = []
    for seg, n in n_series.items():
        method, why = chosen[seg]
        row = {"segment": seg, "n_series": n, "primary_metric": PRIMARY[seg], "method": method,
               "baseline": BASELINE[seg], "reason": why}
        for tag, m in (("method", method), ("baseline", BASELINE[seg])):
            for col in ("wape", "bias", "rmsse"):
                row[f"{tag}_{col}"] = s[col].get((seg, m), np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
