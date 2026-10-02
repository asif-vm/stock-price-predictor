"""Leakage-safe temporal evaluation for the stock forecasting prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "close",
    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_10",
    "sma_ratio_5",
    "sma_ratio_10",
    "sma_ratio_20",
    "volatility_5",
    "volatility_20",
    "volume_change",
]


def build_supervised_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Create causal features at day t and a next-day-close target."""
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float)
    frame = pd.DataFrame(index=prices.index)
    frame["close"] = close
    for lag in (1, 2, 3, 5, 10):
        frame[f"return_{lag}"] = close.pct_change(lag)
    for window in (5, 10, 20):
        frame[f"sma_ratio_{window}"] = close.rolling(window).mean() / close - 1
    daily_return = close.pct_change()
    frame["volatility_5"] = daily_return.rolling(5).std()
    frame["volatility_20"] = daily_return.rolling(20).std()
    frame["volume_change"] = volume.pct_change()
    frame["target_next_close"] = close.shift(-1)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def evaluate_temporal_holdout(
    prices: pd.DataFrame, test_fraction: float = 0.20
) -> dict[str, float | int]:
    """Compare Ridge with a last-observation baseline on the newest rows."""
    frame = build_supervised_frame(prices)
    split = int(len(frame) * (1 - test_fraction))
    if split < 30 or len(frame) - split < 10:
        raise ValueError("At least 50 usable rows are required for evaluation")

    train = frame.iloc[:split]
    test = frame.iloc[split:]
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(train[FEATURE_COLUMNS], train["target_next_close"])

    predictions = model.predict(test[FEATURE_COLUMNS])
    baseline_predictions = test["close"].to_numpy()
    actual = test["target_next_close"].to_numpy()
    model_mae = mean_absolute_error(actual, predictions)
    baseline_mae = mean_absolute_error(actual, baseline_predictions)
    direction_actual = np.sign(actual - baseline_predictions)
    direction_predicted = np.sign(predictions - baseline_predictions)

    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "model_mae": round(float(model_mae), 4),
        "model_rmse": round(float(mean_squared_error(actual, predictions) ** 0.5), 4),
        "baseline_mae": round(float(baseline_mae), 4),
        "baseline_rmse": round(
            float(mean_squared_error(actual, baseline_predictions) ** 0.5), 4
        ),
        "model_mae_change_vs_baseline_pct": round(
            float((baseline_mae - model_mae) / baseline_mae * 100), 2
        ),
        "directional_accuracy": round(
            float((direction_actual == direction_predicted).mean()), 4
        ),
    }


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if data.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def run_benchmark(tickers: list[str], start: str, end: str) -> dict:
    results = {}
    for ticker in tickers:
        prices = download_prices(ticker, start, end)
        results[ticker] = evaluate_temporal_holdout(prices)

    changes = [
        metrics["model_mae_change_vs_baseline_pct"] for metrics in results.values()
    ]
    total_test_rows = sum(metrics["test_rows"] for metrics in results.values())
    return {
        "data_source": "Yahoo Finance adjusted daily OHLCV",
        "period": {"start": start, "end_exclusive": end},
        "split": "oldest 80% train / newest 20% test per ticker",
        "baseline": "predict next close as the current close",
        "model": "StandardScaler + Ridge(alpha=10.0)",
        "tickers": results,
        "summary": {
            "ticker_count": len(results),
            "total_test_rows": total_test_rows,
            "mean_model_mae_change_vs_baseline_pct": round(float(np.mean(changes)), 2),
            "decision": (
                "no-go: retain the naive baseline"
                if np.mean(changes) <= 0
                else "candidate model beats baseline"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"]
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    report = run_benchmark(args.tickers, args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
