import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from evaluation import build_supervised_frame, evaluate_temporal_holdout


def sample_prices(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    changes = rng.normal(0.15, 1.2, rows)
    close = 100 + np.cumsum(changes)
    return pd.DataFrame(
        {
            "Close": close,
            "Volume": rng.integers(100_000, 500_000, rows),
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="B"),
    )


def test_target_is_next_day_close() -> None:
    prices = sample_prices()
    frame = build_supervised_frame(prices)
    first_index = frame.index[0]
    next_index = prices.index[prices.index.get_loc(first_index) + 1]
    assert frame.loc[first_index, "target_next_close"] == prices.loc[next_index, "Close"]


def test_temporal_evaluation_returns_baseline_and_model_metrics() -> None:
    result = evaluate_temporal_holdout(sample_prices())
    assert result["train_rows"] > result["test_rows"] > 0
    assert result["model_mae"] >= 0
    assert result["baseline_mae"] >= 0
    assert 0 <= result["directional_accuracy"] <= 1
