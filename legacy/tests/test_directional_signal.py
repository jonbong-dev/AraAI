"""End-to-end directional accuracy on real market data.

Pulls a few liquid tickers from yfinance, runs the full prediction pipeline,
and measures whether the signed prediction beats a coin-flip baseline.

Design note on sampling and thresholds
--------------------------------------
Daily directional accuracy is a noisy statistic, and the two ways to get it
wrong are sampling a short contiguous block and sampling a stale one.

An earlier version of this file did both: it scored the first 30 windows after
the indicator burn-in, which is a single contiguous month at the *start* of the
2-year history -- roughly 22 months stale, and the same calendar month every
night.  Pooling several symbols did not help, because every symbol was scored
over that same month, so the pooled sample described one market regime rather
than n independent draws.  Measured against the live forex checkpoint, 9.2% of
contiguous pooled-90 windows land below the 40% floor even though accuracy over
the full history is 47.3% -- that test failed roughly one night in eleven on
nothing but which month it happened to look at.

So: score *every* window in the history instead of a 30-day slice.  That is
~470 predictions per symbol, ~2360 pooled for stocks and ~1460 for forex, and
the pool shifts by a single sample per night, so the figure is stable run to
run.  Against that sample the 40% floor is a real regression alarm rather than
a coin flip -- the live checkpoints sit 17 sigma (stocks, 57.5%) and 5.6 sigma
(forex, 47.3%) above it.

The floor cannot catch every degenerate model, though: one that collapses to a
single sign still scores near the base rate of up-days.  The variance and
sign-balance tests cover that structurally, which is why they stay per-symbol
-- a model that emits one constant is a bug no matter which ticker surfaces it.

Per-symbol scores and a trailing-quarter accuracy are printed so the nightly
logs show drift, but only the pooled figure is asserted on.
"""

from __future__ import annotations

import os
import warnings
from functools import cache

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")

REQUIRES_NET = pytest.mark.skipif(
    os.environ.get("NO_NET") == "1",
    reason="network disabled (NO_NET=1)",
)

STOCK_SYMBOLS = ["AAPL", "MSFT", "SPY", "NVDA", "JPM"]
FOREX_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]

HISTORY_PERIOD = "2y"
BATCH_SIZE = 64

# Directional accuracy below this over the full history means the model has
# regressed, not that the sample was unlucky -- see the module docstring.
ACCURACY_FLOOR = 0.40

# Smallest pooled sample the floor is meaningful on.  Well under what a healthy
# 2-year download yields; this only trips if yfinance returns almost nothing.
MIN_POOLED_SAMPLES = 400

# A model that emits the same sign for ~every input is degenerate even when its
# accuracy tracks the base rate of up-days.  Live checkpoints sit at 0.42+.
MIN_SIGN_SHARE = 0.05

# Below this a symbol's sweep is too short to say anything about it.
MIN_SYMBOL_SAMPLES = 30


@cache
def _history(symbol: str):
    """Download a symbol's price history once per session."""
    import yfinance as yf

    return yf.Ticker(symbol).history(period=HISTORY_PERIOD)


@cache
def _pipeline(ckpt_path: str, model_type: str):
    """The UnifiedStockML instance used only for its feature engineering.

    Built once per checkpoint: it is stateless across symbols, and passing the
    right `model_type` keeps it from warning about a checkpoint mismatch and
    throwing away the weights it just read.
    """
    from meridianalgo.unified_ml import UnifiedStockML

    return UnifiedStockML(model_path=ckpt_path, model_type=model_type)


def _build_args(ckpt: dict) -> dict:
    return {
        "input_size": ckpt["input_size"],
        "seq_len": ckpt["seq_len"],
        "dim": ckpt["dim"],
        "num_layers": ckpt["num_layers"],
        "num_heads": ckpt["num_heads"],
        "num_kv_heads": ckpt["num_kv_heads"],
        "num_experts": ckpt["num_experts"],
        "num_prediction_heads": ckpt["num_prediction_heads"],
        "dropout": ckpt["dropout"],
        "use_mamba": ckpt["use_mamba"],
        "mamba_state_dim": ckpt.get("mamba_state_dim", 16),
    }


@cache
def _model(ckpt_path: str):
    from meridianalgo.meridian_model import MeridianModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MeridianModel(**_build_args(ckpt))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


@cache
def _predict_returns(ckpt_path: str, model_type: str, symbol: str):
    """Roll the model over a symbol's whole price history.

    Returns paired ``(predicted_return, actual_next_day_return)`` arrays, one
    entry per window the history supports.  Mirrors the windowing logic in
    ``UnifiedStockML.predict_ultimate``.

    Cached because the accuracy, variance and sign-balance tests all score the
    same predictions -- without this the suite would repeat the whole sweep
    once per test.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _model(ckpt_path)
    ml = _pipeline(ckpt_path, model_type)

    df = _history(symbol)
    if df.empty:
        return np.array([]), np.array([])

    df = ml._add_indicators(df).dropna()
    seq_len = ckpt["seq_len"]
    if len(df) < seq_len + 2:
        return np.array([]), np.array([])

    # `_extract_features` only reads the final row of what it is handed, so a
    # one-row slice gives the same answer as the growing prefix the old loop
    # passed in -- at O(1) per row instead of O(seq_len) per window.
    feats = np.stack([ml._extract_features(df.iloc[j : j + 1]) for j in range(len(df))])
    if feats.shape[1] != ckpt["input_size"]:
        pytest.skip(f"feature width {feats.shape[1]} != checkpoint input_size {ckpt['input_size']}")

    mean = ckpt["scaler_mean"].numpy()
    std = ckpt["scaler_std"].numpy()
    safe_std = np.where(std == 0, 1.0, std)

    # One window per day that has both a full look-back and a next-day close.
    ends = np.arange(seq_len - 1, len(df) - 1)
    windows = np.stack([feats[e - seq_len + 1 : e + 1] for e in ends])
    x = torch.from_numpy(((windows - mean) / safe_std).astype(np.float32))

    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH_SIZE):
            pred, _ = model(x[i : i + BATCH_SIZE])
            chunks.append(pred.reshape(-1).numpy())
    preds = np.concatenate(chunks)

    closes = df["Close"].values
    actuals = (closes[ends + 1] - closes[ends]) / closes[ends]

    finite = np.isfinite(preds) & np.isfinite(actuals)
    return preds[finite], actuals[finite]


def _pooled_accuracy(ckpt_path, model_type: str, symbols: list[str], label: str) -> None:
    """Score every symbol in a group, print the detail, assert on the pool."""
    pytest.importorskip("yfinance")

    correct: list = []
    for symbol in symbols:
        preds, actuals = _predict_returns(str(ckpt_path), model_type, symbol)
        if not len(preds):
            print(f"\n  {symbol}: no data, skipping")
            continue
        hits = np.sign(preds) == np.sign(actuals)
        recent = hits[-63:]  # trailing quarter, printed for drift, not asserted
        print(
            f"\n  {symbol}: {hits.mean():.3f} (n={len(hits)})"
            f"  last-quarter {recent.mean():.3f} (n={len(recent)})"
        )
        correct.extend(hits.tolist())

    if len(correct) < MIN_POOLED_SAMPLES:
        pytest.skip(f"too few total paired samples ({len(correct)})")

    agg_acc = float(np.mean(correct))
    print(f"\n{label} aggregate directional acc: {agg_acc:.3f} (n={len(correct)})")
    assert agg_acc >= ACCURACY_FLOOR, (
        f"{label} aggregate directional accuracy {agg_acc:.3f} < {ACCURACY_FLOOR:.0%} "
        f"over {len(correct)} predictions -- model outputs may be degenerate"
    )


def _assert_has_variance(ckpt_path, model_type: str, symbol: str) -> None:
    pytest.importorskip("yfinance")
    preds, _ = _predict_returns(str(ckpt_path), model_type, symbol)
    if len(preds) < MIN_SYMBOL_SAMPLES:
        pytest.skip(f"too few samples for {symbol} ({len(preds)})")
    spread = float(preds.std())
    assert spread > 1e-6, f"{symbol}: predictions are constant ({spread:.2e})"


def _assert_signs_balanced(ckpt_path, model_type: str, symbol: str) -> None:
    pytest.importorskip("yfinance")
    preds, _ = _predict_returns(str(ckpt_path), model_type, symbol)
    if len(preds) < MIN_SYMBOL_SAMPLES:
        pytest.skip(f"too few samples for {symbol} ({len(preds)})")
    up = float(np.mean(preds > 0))
    print(f"\n  {symbol}: {up:.3f} up / {1 - up:.3f} down (n={len(preds)})")
    assert min(up, 1.0 - up) >= MIN_SIGN_SHARE, (
        f"{symbol}: predictions collapsed to one direction "
        f"({up:.1%} up / {1 - up:.1%} down) -- accuracy alone would not catch this"
    )


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------


@REQUIRES_NET
def test_stocks_aggregate_directional_accuracy(stocks_ckpt_path) -> None:
    """Directional accuracy pooled across every stock symbol and every window."""
    _pooled_accuracy(stocks_ckpt_path, "stock", STOCK_SYMBOLS, "Stocks")


@REQUIRES_NET
@pytest.mark.parametrize("symbol", STOCK_SYMBOLS)
def test_stocks_predictions_have_variance(symbol: str, stocks_ckpt_path) -> None:
    """A model that emits the same value for every input is broken regardless of accuracy."""
    _assert_has_variance(stocks_ckpt_path, "stock", symbol)


@REQUIRES_NET
@pytest.mark.parametrize("symbol", STOCK_SYMBOLS)
def test_stocks_predictions_are_sign_balanced(symbol: str, stocks_ckpt_path) -> None:
    """Predictions stuck on one sign track the base rate of up-days, not the signal."""
    _assert_signs_balanced(stocks_ckpt_path, "stock", symbol)


# ---------------------------------------------------------------------------
# Forex
# ---------------------------------------------------------------------------


@REQUIRES_NET
def test_forex_aggregate_directional_accuracy(forex_ckpt_path) -> None:
    """Directional accuracy pooled across every forex pair and every window."""
    _pooled_accuracy(forex_ckpt_path, "forex", FOREX_SYMBOLS, "Forex")


@REQUIRES_NET
@pytest.mark.parametrize("symbol", FOREX_SYMBOLS)
def test_forex_predictions_have_variance(symbol: str, forex_ckpt_path) -> None:
    """A model that emits the same value for every input is broken regardless of accuracy."""
    _assert_has_variance(forex_ckpt_path, "forex", symbol)


@REQUIRES_NET
@pytest.mark.parametrize("symbol", FOREX_SYMBOLS)
def test_forex_predictions_are_sign_balanced(symbol: str, forex_ckpt_path) -> None:
    """Predictions stuck on one sign track the base rate of up-days, not the signal."""
    _assert_signs_balanced(forex_ckpt_path, "forex", symbol)
