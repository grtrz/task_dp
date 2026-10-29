"""Чтение входных CSV. Никакой логики очистки — только загрузка."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Inputs:
    sales: pd.DataFrame
    stock: pd.DataFrame
    promo: pd.DataFrame
    items: pd.DataFrame


def load_inputs(data_dir: str | Path) -> Inputs:
    d = Path(data_dir)
    return Inputs(
        sales=pd.read_csv(d / "sales_history.csv"),
        stock=pd.read_csv(d / "stock_history.csv"),
        promo=pd.read_csv(d / "promo_calendar.csv"),
        items=pd.read_csv(d / "item_master.csv"),
    )
