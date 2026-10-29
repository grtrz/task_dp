"""Методы прогноза. Единый интерфейс: forecast(y, h) -> np.ndarray длины h.

y — pandas.Series с месячным PeriodIndex без пропусков, последний элемент —
последний известный месяц. Чтобы добавить метод: написать класс с name и forecast,
добавить его в REGISTRY и в CANDIDATES (selection.py) — пайплайн не меняется.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_CMA_W = np.r_[0.5, np.ones(11), 0.5] / 12      # центрированная скользящая средняя 2×12


def seasonal_indices(y: pd.Series, min_months: int = 24, floor: float = 0.1) -> np.ndarray:
    """12 мультипликативных индексов (среднее = 1), позиция 0 — январь.

    Индекс месяца — среднее отношения ряда к скользящей средней 2×12 (по 6 мес. с краёв
    отношения нет). Короче min_months нет даже двух полных циклов — индексы плоские (=1).
    floor защищает от деления на 0, если в какой-то месяц были одни нули.
    """
    idx = np.ones(12)
    if len(y) < min_months:
        return idx
    v = y.to_numpy(float)
    cma = np.convolve(v, _CMA_W, mode="valid")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pd.Series(np.where(cma > 0, v[6:-6] / cma, np.nan), index=y.index[6:-6]).dropna()
    by_month = ratio.groupby(ratio.index.month).mean()
    idx[by_month.index.to_numpy() - 1] = by_month.to_numpy()
    idx = np.maximum(idx, floor)
    return idx / idx.mean()


class Naive:
    name = "naive"

    def forecast(self, y: pd.Series, h: int) -> np.ndarray:
        return np.full(h, y.iloc[-1], float)


class SeasonalNaive:
    """Тот же месяц год назад. Короче года — откат к naive."""
    name = "snaive"

    def forecast(self, y: pd.Series, h: int) -> np.ndarray:
        if len(y) < 12:
            return Naive().forecast(y, h)
        return np.resize(y.to_numpy(float)[-12:], h)


def _ses_level(x: np.ndarray, alphas=(0.1, 0.2, 0.3, 0.5)) -> float:
    """Уровень простого экспоненциального сглаживания; alpha — по SSE однопошаговых ошибок."""
    best_sse, best_level = np.inf, x[-1]
    for a in alphas:
        level, sse = x[0], 0.0
        for v in x[1:]:
            sse += (v - level) ** 2
            level = a * v + (1 - a) * level
        if sse < best_sse:
            best_sse, best_level = sse, level
    return float(best_level)


class SeasonalSES:
    """Сезонные индексы (отношение к MA) × уровень SES по десезонированному ряду."""
    name = "ses_seasonal"

    def forecast(self, y: pd.Series, h: int) -> np.ndarray:
        idx = seasonal_indices(y)
        level = _ses_level(y.to_numpy(float) / idx[y.index.month.to_numpy() - 1])
        months = pd.period_range(y.index[-1] + 1, periods=h, freq="M").month.to_numpy()
        return level * idx[months - 1]


class SES:
    name = "ses"

    def forecast(self, y: pd.Series, h: int) -> np.ndarray:
        return np.full(h, _ses_level(y.to_numpy(float)))


class MeanLast:
    def __init__(self, n: int):
        self.n, self.name = n, f"mean{n}"

    def forecast(self, y: pd.Series, h: int) -> np.ndarray:
        return np.full(h, y.to_numpy(float)[-self.n:].mean())


class CrostonSBA:
    """Croston с поправкой Сынтетоса–Бойлана: оценка размера и интервала спроса отдельно."""
    name = "croston_sba"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def forecast(self, y: pd.Series, h: int) -> np.ndarray:
        v = y.to_numpy(float)
        nz = np.flatnonzero(v > 0)
        if len(nz) == 0:
            return np.zeros(h)
        z, p, gap = v[nz[0]], float(nz[0] + 1), 1
        for x in v[nz[0] + 1:]:
            if x > 0:
                z = self.alpha * x + (1 - self.alpha) * z
                p = self.alpha * gap + (1 - self.alpha) * p
                gap = 1
            else:
                gap += 1
        return np.full(h, (1 - self.alpha / 2) * z / p)


REGISTRY = {m.name: m for m in (Naive(), SeasonalNaive(), SeasonalSES(), SES(), MeanLast(3), MeanLast(12), CrostonSBA())}


def predict(name: str, y: pd.Series, h: int) -> np.ndarray:
    """Прогноз метода из REGISTRY; спрос не бывает отрицательным, поэтому нижняя граница — 0."""
    return np.maximum(REGISTRY[name].forecast(y, h), 0.0)
