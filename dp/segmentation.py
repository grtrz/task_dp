"""Сегментация рядов по правилам (порядок проверок важен — первое сработавшее побеждает).

  short         — меньше 24 мес.: нет двух полных сезонных циклов, сезонность не оценить
  intermittent  — ADI > 1.32 (порог Сынтетоса–Бойлана): много нулей, MAPE не определена
  regular       — всё остальное

Сезонность в правило не входит: её сила на 24 точках оценивается шумно (у чистого шума
скорректированный R² доходит до 0.5), поэтому сезонный или несезонный метод выбирает бэктест.
"""
from __future__ import annotations

import pandas as pd

from .cleaning import KEY

MIN_MONTHS = 24
ADI_THRESHOLD = 1.32


def classify(y: pd.Series) -> str:
    nonzero = int((y > 0).sum())
    if len(y) < MIN_MONTHS:
        return "short"
    if nonzero == 0 or len(y) / nonzero > ADI_THRESHOLD:
        return "intermittent"
    return "regular"


def segment_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Индекс (sku, location); колонки n_months, segment."""
    g = panel.groupby(KEY)["qty"]
    return pd.DataFrame({"n_months": g.size(), "segment": g.apply(classify)})
