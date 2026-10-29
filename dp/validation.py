"""Бэктест (rolling origin) и метрики.

Фолды: три непересекающихся квартала 2025 года (origin = конец истории − 9/6/3 мес.), горизонт 3,
как в боевом прогнозе. Сравниваем только с НАБЛЮДАЕМЫМИ месяцами (observed): промо, дефицит и
подставленные значения в «факт» не попадают — иначе мы мерили бы себя по собственной очистке.

Метрики сегмента. Ошибка каждой точки делится на средний уровень ряда в обучении (mu), и только
потом точки копятся в сегмент: иначе SKU в «тысячекратных» единицах (SKU_0018) заглушил бы остальные,
а у прерывистых рядов Σфакта в окне почти ноль и относительные ошибки взрываются.
  wape  = Σ|f-a|/mu / Σ a/mu      — доля объёма, определена при нулях (в отличие от MAPE)
  bias  = Σ(f-a)/mu / Σ a/mu      — знак: >0 завышение (склад), <0 занижение (дефицит)
  rmsse = sqrt(mean(e² / scale))  — scale = средний квадрат разности соседних месяцев в обучении;
                                    для прерывистых рядов, где MAE-метрики поощряют нулевой прогноз
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import KEY
from .methods import predict


def build_series(panel: pd.DataFrame) -> dict[tuple, tuple[pd.Series, pd.Series]]:
    """{(sku, location): (qty, observed)} — оба с PeriodIndex."""
    out = {}
    for key, d in panel.groupby(KEY):
        d = d.set_index("period")
        out[key] = (d["qty"], d["observed"])
    return out


def _one_fold(y: pd.Series, obs: pd.Series, origin: pd.Period, h: int, method: str, window: int | None = None):
    train = y[:origin]
    if window:
        train = train.iloc[-window:]
    future = pd.period_range(origin + 1, periods=h, freq="M")
    ok = obs.reindex(future, fill_value=False).to_numpy(bool)
    if not ok.any():
        return None
    f = predict(method, train, h)
    mu, scale = float(train.mean()), float(np.mean(np.diff(train.to_numpy(float)) ** 2))
    return pd.DataFrame({
        "period": future[ok], "actual": y.reindex(future).to_numpy(float)[ok], "forecast": f[ok],
        "mu": mu if mu > 0 else np.nan, "scale": scale if scale > 0 else np.nan,
    })


def backtest(series, seg_of: dict, candidates: dict[str, list[str]], origins, h: int,
             min_train: int = 24, window: int | None = None) -> pd.DataFrame:
    """Все кандидаты сегмента на всех фолдах каждого ряда. seg_of: {(sku, location): сегмент};
    ряды вне seg_of пропускаются. window обрезает обучение до последних N месяцев."""
    parts = []
    for key, seg in seg_of.items():
        y, obs = series[key]
        for origin in origins:
            if len(y[:origin]) < min_train:
                continue
            for m in candidates[seg]:
                r = _one_fold(y, obs, origin, h, m, window)
                if r is not None:
                    parts.append(r.assign(sku=key[0], location=key[1], segment=seg, method=m, origin=origin))
    return pd.concat(parts, ignore_index=True)


def segment_scores(bt: pd.DataFrame) -> pd.DataFrame:
    bt = bt[bt["mu"].notna()]
    err = bt["forecast"] - bt["actual"]
    n = pd.DataFrame({"abs_err": err.abs() / bt["mu"], "err": err / bt["mu"],
                      "actual": bt["actual"] / bt["mu"], "sq_err": err ** 2 / bt["scale"]})
    g = n.groupby([bt["segment"], bt["method"]])
    total = g["actual"].sum().where(lambda s: s > 0)
    return pd.DataFrame({
        "wape": g["abs_err"].sum() / total,
        "bias": g["err"].sum() / total,
        "rmsse": np.sqrt(g["sq_err"].mean()),
        "n_series": bt.drop_duplicates(["segment", "method"] + KEY).groupby(["segment", "method"]).size(),
    }).reset_index()
