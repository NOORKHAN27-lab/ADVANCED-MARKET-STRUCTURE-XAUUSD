"""
filters.py
----------
Extra selectivity filters that turn "every detected setup" into "only the
setups matching your actual execution rules":

1. Session filter — only take setups whose entry/confirmation bar falls
   within the New York session (default 08:00-17:00 America/New_York,
   DST-aware).
2. Fresh-zone filter — only take a supply/demand zone the *first* time
   price returns to it; if the same zone would be tested again by a later
   setup, that later one is dropped.
3. Risk:Reward filter — only keep a setup if at least one of its targets
   produces a reward:risk ratio within [min_rr, max_rr]; picks the closer
   qualifying target.
"""

from typing import List, Literal, Optional, Tuple

import pandas as pd


def in_ny_session(timestamp, start_hour: int = 8, end_hour: int = 17) -> bool:
    """True if `timestamp` (any tz or naive, assumed UTC if naive) falls
    within the NY session window on the America/New_York local clock."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ny_time = ts.tz_convert("America/New_York")
    return start_hour <= ny_time.hour < end_hour


def zones_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """True if price ranges a=(low,high) and b=(low,high) intersect at all."""
    return not (a[1] < b[0] or b[1] < a[0])


def select_target_by_rr(entry_zone: Tuple[float, float], stop_loss: float,
                         targets: List[float], direction: Literal["BUY", "SELL"],
                         min_rr: float = 3.0, max_rr: float = 5.0) -> Optional[Tuple[float, float]]:
    """
    Returns (target_price, risk_reward_ratio) for the closest target whose
    R:R falls within [min_rr, max_rr], or None if no target qualifies.
    """
    entry_price = (entry_zone[0] + entry_zone[1]) / 2
    risk = abs(entry_price - stop_loss)
    if risk <= 0:
        return None

    qualifying = []
    for t in targets:
        reward = (t - entry_price) if direction == "BUY" else (entry_price - t)
        if reward <= 0:
            continue
        rr = reward / risk
        if min_rr <= rr <= max_rr:
            qualifying.append((t, rr))

    if not qualifying:
        return None
    # prefer the closer/lower-RR qualifying target (more conservative, more likely to hit)
    qualifying.sort(key=lambda x: x[1])
    return qualifying[0]


def find_valid_retracement_entry(data: pd.DataFrame, entry_zone: Tuple[float, float],
                                  start_index: int, direction: Literal["BUY", "SELL"],
                                  max_lookahead: int = 60) -> Optional[int]:
    """
    Scans forward from `start_index` for a genuine, progressive pullback
    into `entry_zone`:
      - BUY: the retracement must be made of RED (bearish) candles --
        at least 2 consecutive -- where each new red candle's close
        extends further down than the previous one (real follow-through,
        not chop).
      - SELL: mirror image -- GREEN (bullish) candles, at least 2
        consecutive, each closing further up than the previous one.

    Returns the index of the candle where this valid run's close lands
    inside `entry_zone` (the actual entry bar), or None if no such
    pattern is found within `max_lookahead` bars.
    """
    low, high = entry_zone
    opens = data["Open"].to_numpy()
    closes = data["Close"].to_numpy()
    end = min(start_index + max_lookahead, len(data))

    run_length = 0
    prev_close = None

    for i in range(start_index, end):
        is_counter_candle = (closes[i] < opens[i]) if direction == "BUY" else (closes[i] > opens[i])

        if is_counter_candle:
            if run_length == 0:
                run_length = 1
            else:
                extends = (closes[i] < prev_close) if direction == "BUY" else (closes[i] > prev_close)
                run_length = run_length + 1 if extends else 1
            prev_close = closes[i]

            if run_length >= 2 and low <= closes[i] <= high:
                return i
        else:
            run_length = 0
            prev_close = None

    return None
