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
