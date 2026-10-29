"""Правило выбора метода на СЕГМЕНТ (не на SKU — на 3 фолдах выбор по ряду переобучается).

Правило, одинаковое для всех сегментов:
  1. Кандидаты со смещением |bias| > bias_cap исключаются (смещение дороже случайной ошибки),
     baseline исключению не подлежит — он точка отсчёта.
  2. Среди остальных берём минимум основной метрики сегмента.
  3. Победитель заменяет baseline, только если лучше его не менее чем на min_gain (относительно).
     Иначе остаётся baseline: более сложный метод должен доказать преимущество.
  4. Если в сегменте меньше min_series рядов с посчитанной метрикой — остаётся baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CANDIDATES = {
    "regular": ["snaive", "ses_seasonal", "ses"],
    "intermittent": ["snaive", "mean12", "croston_sba"],
    "short": ["naive", "mean3", "ses"],
}
BASELINE = {"regular": "snaive", "intermittent": "snaive", "short": "naive"}
PRIMARY = {"regular": "wape", "intermittent": "rmsse", "short": "wape"}


@dataclass(frozen=True)
class Rule:
    bias_cap: float = 0.10
    min_gain: float = 0.05
    min_series: int = 3


def select_method(scores: pd.DataFrame, segment: str, rule: Rule = Rule()) -> tuple[str, str]:
    base, metric = BASELINE[segment], PRIMARY[segment]
    s = scores[scores["segment"] == segment].set_index("method")
    if base not in s.index or s.loc[base, "n_series"] < rule.min_series:
        return base, "мало рядов для сравнения: оставлен baseline"
    ok = s[(s["bias"].abs() <= rule.bias_cap) | (s.index == base)]
    best = ok[metric].idxmin()
    if best == base:
        return base, "baseline не побит"
    gain = 1 - ok.loc[best, metric] / s.loc[base, metric]
    if not gain >= rule.min_gain:
        return base, f"{best} лучше baseline лишь на {gain:.1%} (< {rule.min_gain:.0%}): оставлен baseline"
    return best, f"{best} лучше baseline на {gain:.1%} по {metric}"
