"""python -m dp [--data data] [--out output]"""
from __future__ import annotations

import argparse

from .pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="DP: очистка истории, бэктест и прогноз на 3 месяца")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    forecast, summary, log = run(args.data, args.out)
    print("Журнал очистки:")
    print(log.groupby("step").size().to_string())
    print("\nСегменты и методы:")
    print(summary.round(3).to_string(index=False))
    print(f"\nПрогноз: {args.out}/forecast.csv ({len(forecast)} строк)")


if __name__ == "__main__":
    main()
