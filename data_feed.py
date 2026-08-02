"""
data_feed.py
------------
Fetches OHLC candle data for XAUUSD (Gold) via yfinance.

Returns a pandas DataFrame indexed by datetime with columns:
    Open, High, Low, Close, Volume

If network access or yfinance is unavailable, falls back to deterministic
synthetic data so the rest of the app can still be demoed/tested offline.
"""

from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# yfinance interval strings differ slightly
YFINANCE_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "1h", "1d": "1d",
    # yfinance has no native 4h bar for most tickers; we resample 1h -> 4h.
}


def get_xauusd(interval: str = "1h", lookback_days: int = 60) -> pd.DataFrame:
    """Fetch XAUUSD (Gold futures proxy GC=F) candles via yfinance."""
    # yfinance intraday history limits: 1m ~7d, 5m/15m/30m ~60d, 1h ~730d
    max_days_by_interval = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 729, "1d": 3650}
    if YFINANCE_AVAILABLE:
        try:
            yf_interval = YFINANCE_INTERVAL_MAP.get(interval, "1h")
            capped_days = min(lookback_days, max_days_by_interval.get(yf_interval, 60))
            period = f"{capped_days}d"
            data = yf.download("GC=F", period=period, interval=yf_interval, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            if len(data) > 0:
                if interval == "4h" and yf_interval == "1h":
                    data = _resample(data, "4h")
                return data[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass
    return get_synthetic_candles(interval=interval, seed=7, start_price=2000.0)


def _resample(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return data.resample(rule).agg(agg).dropna()


def get_synthetic_candles(interval: str = "1h", num_bars: int = 500,
                           start_price: float = 2000.0, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic OHLC data with realistic swing structure (used
    when live data isn't reachable — e.g. offline development/demo).
    """
    rng = np.random.default_rng(seed)
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(interval, "1h")

    dates = pd.date_range(end=datetime.now(), periods=num_bars, freq=freq)
    closes = [start_price]
    # random walk with occasional trending "impulse" legs so structure looks real
    trend_bias = 0.0
    for i in range(1, num_bars):
        if i % 40 == 0:
            trend_bias = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        change = rng.normal(loc=trend_bias, scale=start_price * 0.004)
        closes.append(max(closes[-1] + change, 1.0))

    closes = np.array(closes)
    highs = closes + np.abs(rng.normal(loc=start_price * 0.0015, scale=start_price * 0.001, size=num_bars))
    lows = closes - np.abs(rng.normal(loc=start_price * 0.0015, scale=start_price * 0.001, size=num_bars))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    volumes = rng.uniform(100, 1000, size=num_bars)

    return pd.DataFrame({
        "Open": opens, "High": np.maximum(highs, np.maximum(opens, closes)),
        "Low": np.minimum(lows, np.minimum(opens, closes)),
        "Close": closes, "Volume": volumes,
    }, index=dates)
