# DP: прогноз на грязной истории

Прогноз спроса на 2026-01…03 по `sku × location`: очистка → сегментация → бэктест → выбор метода на сегмент → прогноз.
Подробности и обоснования — в [REPORT.md](REPORT.md).

## Запуск (Python 3.10+)

```bash
pip install -r requirements.txt && python -m dp
```

Результат — `output/forecast.csv` (`sku, location, period, forecast, segment, method`). Рядом в `output/`:
`cleaning_log.csv` (каждое исправление данных), `clean_history.csv`, `backtest.csv`, `scores.csv`,
`segment_report.csv`. Параметры: `python -m dp --data data --out output`.

## Тесты

```bash
python -m unittest discover -s tests -t .
```

`tests/test_cleaning.py` — по тесту на каждый дефект данных + сквозной прогон на реальных файлах;
`tests/test_forecasting.py` — сегментация, методы, отсутствие утечки в бэктесте, правило выбора, пайплайн целиком.

## Структура

| Модуль | Роль |
|---|---|
| `dp/io.py` | чтение входных CSV |
| `dp/cleaning.py` | очистка, каждый шаг пишет в журнал |
| `dp/segmentation.py` | правила сегментации (`short` / `intermittent` / `regular`) |
| `dp/methods.py` | методы с общим интерфейсом `forecast(y, h)`, `REGISTRY`, сезонные индексы |
| `dp/validation.py` | rolling-origin бэктест, метрики |
| `dp/selection.py` | кандидаты по сегментам, правило выбора |
| `dp/pipeline.py` | сборка всего вместе |

Новый метод: класс в `methods.py`, строка в `REGISTRY`, имя в `CANDIDATES` (`selection.py`).
