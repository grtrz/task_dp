"""Очистка истории продаж.

Каждый шаг — функция «панель -> панель», все исправления пишутся в CleaningLog:
молча менять данные нельзя, журнал идёт в REPORT.md и проверяется тестами.

Панель — длинная таблица с полным месячным календарём на каждый ряд (sku, location):
    qty_raw           — как пришло из источника (NaN для пропущенных месяцев)
    qty               — очищенное значение (NaN = «не знаем», до шага impute)
    flag              — причина, по которой значение было заменено ('' если не трогали)
    from_predecessor  — строка подставлена из истории предшественника (sku→sku)
    observed          — реальное наблюдение этого SKU; только такие месяцы идут в «факт» бэктеста

Порядок шагов зафиксирован в clean() и важен: единицы измерения ищем уже без
промо-пиков и нулей от дефицита, склейку делаем после выравнивания единиц,
подстановку — последней, когда известны все «дыры».
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import Inputs

KEY = ["sku", "location"]


@dataclass(frozen=True)
class CleaningConfig:
    oos_min_days: int = 15        # месяц с >= N днями без товара: продажи цензурированы
    unit_log_tol: float = 0.15    # допуск на |log10(отношение) - целая степень 10|
    window: int = 3               # месяцев: окно медиан для разрыва единиц, склейки и подстановки


class CleaningLog:
    COLUMNS = ["step", "sku", "location", "period", "detail"]

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, step: str, sku, location, period, detail: str) -> None:
        self._rows.append({"step": step, "sku": sku, "location": location, "period": str(period), "detail": detail})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=self.COLUMNS)

    def count(self, step: str) -> int:
        return sum(r["step"] == step for r in self._rows)


def _to_period(s: pd.Series, what: str) -> pd.Series:
    try:
        return pd.to_datetime(s, format="%Y-%m").dt.to_period("M")
    except (ValueError, TypeError) as e:
        raise ValueError(f"{what}: период не в формате YYYY-MM ({e})") from e


# --- 1. дубли ----------------------------------------------------------------

def drop_duplicates(sales: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Полный дубль — повтор загрузки, удаляем (не суммируем).

    Одинаковый ключ с разным qty — конфликт: оставляем последнюю запись
    (позднюю загрузку), в журнале помечаем отдельно.
    """
    exact = sales.duplicated(keep="first")
    for r in sales[exact].itertuples():
        log.add("duplicate_exact", r.sku, r.location, r.period, f"qty={r.qty}: полный дубль строки, удалён")
    sales = sales[~exact]

    conflict = sales.duplicated(KEY + ["period"], keep="last")
    for r in sales[conflict].itertuples():
        log.add("duplicate_conflict", r.sku, r.location, r.period, f"qty={r.qty}: другое значение на тот же ключ, оставлена последняя запись")
    return sales[~conflict].reset_index(drop=True)


# --- 2. календарь ------------------------------------------------------------

def _complete_calendar(d: pd.DataFrame) -> pd.DataFrame:
    """Один ряд -> без пропущенных месяцев между первым и последним наблюдением."""
    sku, loc = d["sku"].iat[0], d["location"].iat[0]
    idx = pd.period_range(d["period"].min(), d["period"].max(), freq="M")
    out = d.set_index("period").reindex(idx).rename_axis("period").reset_index()
    out["sku"], out["location"] = sku, loc
    out["flag"] = out["flag"].fillna("")
    out["from_predecessor"] = out["from_predecessor"].eq(True)
    out.loc[out["qty"].isna() & (out["flag"] == ""), "flag"] = "gap"
    return out


def build_panel(sales: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Явный календарь: отсутствующая строка — это NaN, а не 0 и не «ничего».

    Календарь ряда — от первого до последнего его наблюдения: ряд, который
    закончился раньше конца истории, не дотягиваем NaN-ами до 2025-12.
    """
    s = sales.copy()
    s["period"] = _to_period(s["period"], "sales")
    s["qty_raw"] = s["qty"].astype(float)
    s["qty"] = s["qty_raw"]
    s["flag"] = ""
    s["from_predecessor"] = False
    panel = pd.concat([_complete_calendar(d) for _, d in s.groupby(KEY)], ignore_index=True)
    for r in panel[panel["flag"] == "gap"].itertuples():
        log.add("gap", r.sku, r.location, r.period, "месяца нет в данных (≠ ноль продаж), помечен как пропуск")
    return panel


# --- 3. маски: значения, которым нельзя доверять -----------------------------

def _mask(panel: pd.DataFrame, sel, flag: str, log: CleaningLog, detail) -> pd.DataFrame:
    panel = panel.copy()
    sel = np.asarray(sel) & panel["qty"].notna().to_numpy() & (panel["flag"] == "").to_numpy()
    for r in panel[sel].itertuples():
        log.add(flag, r.sku, r.location, r.period, detail(r))
    panel.loc[sel, "qty"] = np.nan
    panel.loc[sel, "flag"] = flag
    return panel


def mask_negatives(panel: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Отрицательные продажи — возврат или сторно, не спрос. Считаем пропуском, не нулём."""
    return _mask(panel, panel["qty"] < 0, "negative", log,
                 lambda r: f"qty={r.qty:g}: отрицательные продажи, значение отброшено")


def mask_stockouts(panel: pd.DataFrame, stock: pd.DataFrame, cfg: CleaningConfig, log: CleaningLog) -> pd.DataFrame:
    """Нули в дефиците — не отсутствие спроса. Обучаться на них нельзя."""
    st = stock.assign(period=_to_period(stock["period"], "stock")).drop_duplicates(KEY + ["period"], keep="last")
    days = panel[KEY + ["period"]].merge(st[KEY + ["period", "days_out_of_stock"]], how="left", on=KEY + ["period"])
    return _mask(panel, days["days_out_of_stock"].fillna(0).to_numpy() >= cfg.oos_min_days, "oos", log,
                 lambda r: f"qty={r.qty:g}: месяц в дефиците, продажи цензурированы")


def mask_promo(panel: pd.DataFrame, promo: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Промо-месяц — разовый пик (в данных ~4×). Иначе повторяющаяся акция станет «сезонностью»."""
    keys = pd.MultiIndex.from_arrays([promo["sku"], _to_period(promo["period"], "promo")])
    sel = pd.MultiIndex.from_frame(panel[["sku", "period"]]).isin(keys)
    return _mask(panel, sel, "promo", log, lambda r: f"qty={r.qty:g}: промо-месяц, значение заменяется базовым уровнем")


# --- 4. единицы измерения ----------------------------------------------------

def _decade(ratio: float, tol: float) -> int:
    """Целая степень десяти, к которой близко отношение (0 — если не близко)."""
    k = np.log10(ratio)
    p = int(round(k))
    return p if abs(k - p) <= tol else 0


def find_unit_breaks(values, cfg: CleaningConfig = CleaningConfig()) -> list[tuple[int, float]]:
    """Скачок уровня в целую степень десяти, подтверждённый медианами до/после.

    Скачок между соседними наблюдениями проверяем по 10^k (допуск unit_log_tol),
    затем требуем то же по медианам окон: одиночный пик (промо, выброс) отсекается,
    потому что медиана «после» его не видит. Возвращает [(индекс первого месяца
    после разрыва, множитель after/before)].
    """
    v = np.asarray(values, float)
    w, out, last = cfg.window, [], None
    for t in range(len(v)):
        if not np.isfinite(v[t]):
            continue
        if last is not None and v[t] > 0 and v[last] > 0:
            p = _decade(v[t] / v[last], cfg.unit_log_tol)
            if p:
                before, after = v[max(0, t - w):t], v[t:t + w]
                before, after = before[np.isfinite(before)], after[np.isfinite(after)]
                if min(len(before), len(after)) >= 2:
                    mb, ma = np.median(before), np.median(after)
                    if mb > 0 and ma > 0 and _decade(ma / mb, cfg.unit_log_tol) == p:
                        out.append((t, 10.0 ** p))
        last = t
    return out


def fix_unit_breaks(panel: pd.DataFrame, cfg: CleaningConfig, log: CleaningLog) -> pd.DataFrame:
    """Приводим историю к ТЕКУЩИМ единицам учётной системы (последний сегмент ряда).

    Прогноз уходит в планирование поставок и должен совпадать с тем, что система
    показывает сейчас. Обратное направление — одна строка (множитель), см. REPORT.
    """
    panel = panel.copy()
    for (sku, loc), g in panel.groupby(KEY):
        v = g["qty"].to_numpy().copy()
        for t, f in find_unit_breaks(v, cfg):
            v[:t] *= f
            log.add("unit_break", sku, loc, g["period"].iat[t],
                    f"скачок уровня ×{f:g} с этого месяца; история до него умножена на {f:g} (приведена к текущим единицам)")
        panel.loc[g.index, "qty"] = v
    return panel


# --- 5. преемственность SKU --------------------------------------------------

def stitch_predecessors(panel: pd.DataFrame, items: pd.DataFrame, cfg: CleaningConfig, log: CleaningLog) -> pd.DataFrame:
    """История предшественника достраивается перед стартом преемника.

    Уровень предшественника масштабируем отношением средних на границе
    (первые N месяцев преемника / последние N месяцев предшественника).
    Строки помечаем from_predecessor — в оценке качества они не участвуют.
    """
    pairs = items.dropna(subset=["predecessor_sku"])[["sku", "predecessor_sku"]]
    for succ, pre in pairs.itertuples(index=False):
        for loc in panel.loc[panel["sku"] == succ, "location"].unique():
            own = panel[(panel["sku"] == succ) & (panel["location"] == loc)]
            old = panel[(panel["sku"] == pre) & (panel["location"] == loc)]
            first = own["period"].min()
            hist = old[old["period"] < first]
            if hist.empty:
                continue
            tail = old["qty"].to_numpy()[old["qty"].notna().to_numpy()][-cfg.window:]
            head = own["qty"].to_numpy()[own["qty"].notna().to_numpy()][:cfg.window]
            scale = head.mean() / tail.mean() if len(tail) and len(head) and tail.mean() > 0 else 1.0
            hist = hist.assign(sku=succ, qty=hist["qty"] * scale, from_predecessor=True)
            joined = _complete_calendar(pd.concat([hist, own], ignore_index=True))
            panel = pd.concat([panel.drop(own.index), joined], ignore_index=True)
            log.add("stitch", succ, loc, first, f"добавлено {len(hist)} мес. истории {pre} (масштаб ×{scale:.3f} по границе)")
    return panel.sort_values(KEY + ["period"]).reset_index(drop=True)


def drop_ended_series(panel: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Ряд, закончившийся раньше конца истории (снят с продажи или влит в преемника), не прогнозируем:
    прогноз от его последнего месяца сдвинул бы периоды."""
    ended = (panel.groupby(KEY)["period"].transform("max") < panel["period"].max()).to_numpy()
    for (sku, loc), g in panel[ended].groupby(KEY):
        log.add("series_ended", sku, loc, g["period"].max(), "ряд закончился раньше конца истории, в прогноз не идёт")
    return panel[~ended].reset_index(drop=True)


# --- 6. подстановка ----------------------------------------------------------

def _fill(s: pd.Series, window: int) -> pd.Series:
    near = s.rolling(2 * window + 1, center=True, min_periods=1).median()
    return s.fillna(near).ffill().bfill()


def impute(panel: pd.DataFrame, cfg: CleaningConfig, log: CleaningLog) -> pd.DataFrame:
    """NaN -> медиана реальных значений в окне ±window (NaN в окне не считаются).

    Подставленное не питает следующую подстановку: медиана берётся по исходным
    значениям. Медиана, а не среднее: соседи могут содержать пики. Если в окне
    пусто — ближайшее известное значение.
    """
    panel = panel.copy()
    filled = panel.groupby(KEY)["qty"].transform(_fill, cfg.window)
    gap = panel["qty"].isna()
    for r in panel[gap].assign(fill=filled[gap]).itertuples():
        log.add("impute", r.sku, r.location, r.period, f"{r.flag}: подставлено {r.fill:.1f} (медиана соседей)")
    panel["qty"] = filled
    return panel


def clean(inp: Inputs, cfg: CleaningConfig = CleaningConfig()) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog()
    panel = build_panel(drop_duplicates(inp.sales, log), log)
    panel = mask_negatives(panel, log)
    panel = mask_stockouts(panel, inp.stock, cfg, log)
    panel = mask_promo(panel, inp.promo, log)
    panel = fix_unit_breaks(panel, cfg, log)
    panel = stitch_predecessors(panel, inp.items, cfg, log)
    panel = drop_ended_series(panel, log)
    panel = impute(panel, cfg, log)
    panel["observed"] = (panel["flag"] == "") & ~panel["from_predecessor"]
    return panel, log
