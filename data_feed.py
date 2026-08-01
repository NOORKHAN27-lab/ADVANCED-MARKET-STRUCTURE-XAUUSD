"""
data_feed.py
------------
Fetches OHLC candle data for XAUUSD (via yfinance) and BTCUSD (via the
Binance public REST API — no API key required for market data).

Both fetchers return a pandas DataFrame indexed by datetime with columns:
    Open, High, Low, Close, Volume

If network access or a dependency is unavailable, both fall back to
deterministic synthetic data so the rest of the app can still be
demoed/tested offline.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Binance interval strings: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
BINANCE_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d",
}

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


def get_btcusd(interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    Fetch BTCUSDT candles via the Binance public klines endpoint.
    `limit` can exceed Binance's single-request cap of 1000 -- this
    paginates backward with multiple requests to build up a much longer
    history (e.g. months of 15m data), so old liquidity levels are still
    present for sweep detection, not just the last ~10 days.
    """
    if not REQUESTS_AVAILABLE:
        return get_synthetic_candles(interval=interval, seed=99, start_price=65000.0)

    binance_interval = BINANCE_INTERVAL_MAP.get(interval, "1h")
    per_request = 1000
    all_frames = []
    end_time = None  # ms epoch; None = most recent

    try:
        remaining = limit
        while remaining > 0:
            batch_size = min(per_request, remaining)
            params = {"symbol": "BTCUSDT", "interval": binance_interval, "limit": batch_size}
            if end_time is not None:
                params["endTime"] = end_time

            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                break

            df = pd.DataFrame(raw, columns=[
                "open_time", "Open", "High", "Low", "Close", "Volume",
                "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
            ])
            all_frames.append(df)

            if len(raw) < batch_size:
                break  # reached the beginning of available history

            end_time = int(raw[0][0]) - 1  # next batch ends just before this batch's first candle
            remaining -= batch_size

        if not all_frames:
            return get_synthetic_candles(interval=interval, seed=99, start_price=65000.0)

        combined = pd.concat(all_frames, ignore_index=True)
        combined["open_time"] = pd.to_datetime(combined["open_time"], unit="ms")
        combined = combined.drop_duplicates(subset="open_time").sort_values("open_time")
        combined = combined.set_index("open_time")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            combined[col] = combined[col].astype(float)
        return combined[["Open", "High", "Low", "Close", "Volume"]]

    except Exception:
        return get_synthetic_candles(interval=interval, seed=99, start_price=65000.0)


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
