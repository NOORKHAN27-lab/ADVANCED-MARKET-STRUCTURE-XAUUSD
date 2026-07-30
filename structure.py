"""
structure.py
------------
Detects swing highs/lows (fractal pivots) and labels market structure
breaks as BOS (Break of Structure — trend continuation) or CHOCH
(Change of Character — trend reversal), the same vocabulary used in
Smart-Money/ICT-style market structure analysis.

Algorithm
---------
1. Swing points: a bar `i` is a swing high if its High is the max of the
   `left` bars before and `right` bars after it (and similarly for swing
   lows). This means a swing is only confirmed `right` bars later — a
   standard, non-repainting fractal definition.

2. Structure state machine: walk forward bar-by-bar. Track the most
   recent *unbroken* swing high and swing low. When price closes beyond
   one of them:
     - If that break is in the same direction as the current trend →
       label it BOS (continuation).
     - If it's in the opposite direction (or there is no trend yet) →
       label it CHOCH (reversal) and flip the trend.
   After a break, the broken level is retired and the next fresh swing
   in that direction becomes the new level being watched.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd

Bias = Literal["bullish", "bearish", "neutral"]


@dataclass
class SwingPoint:
    index: int
    date: object
    price: float
    kind: Literal["high", "low"]


@dataclass
class StructureEvent:
    index: int
    date: object
    price: float
    label: Literal["BOS_bull", "BOS_bear", "CHOCH_bull", "CHOCH_bear"]


def find_swings(data: pd.DataFrame, left: int = 2, right: int = 2) -> List[SwingPoint]:
    """Detect fractal swing highs and lows in `data` (needs High/Low columns)."""
    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()
    n = len(data)
    swings: List[SwingPoint] = []

    for i in range(left, n - right):
        window_high = highs[i - left: i + right + 1]
        window_low = lows[i - left: i + right + 1]

        if highs[i] == window_high.max() and (window_high == highs[i]).sum() == 1:
            swings.append(SwingPoint(index=i, date=data.index[i], price=highs[i], kind="high"))
        if lows[i] == window_low.min() and (window_low == lows[i]).sum() == 1:
            swings.append(SwingPoint(index=i, date=data.index[i], price=lows[i], kind="low"))

    swings.sort(key=lambda s: s.index)
    return swings


def label_structure(data: pd.DataFrame, swings: List[SwingPoint]) -> List[StructureEvent]:
    """
    Walk forward through candles and label BOS/CHOCH breaks.

    Confirmation rule (matches manual chart-reading strategy):
      - BOS (continuation, same direction as current trend): confirmed by
        ONE candle closing beyond the level, body close (not wick).
      - CHOCH (reversal, against current trend): requires TWO CONSECUTIVE
        candle body-closes beyond the level before the structure shift is
        confirmed. A single close beyond the level that isn't followed by
        a second consecutive close is NOT a confirmed CHOCH -- the level
        stays "watched" and unbroken.
    """
    events: List[StructureEvent] = []
    trend: Optional[Bias] = None

    watched_high: Optional[SwingPoint] = None
    watched_low: Optional[SwingPoint] = None
    swing_i = 0
    closes = data["Close"].to_numpy()

    pending_choch_bull_at: Optional[int] = None  # index of the 1st (unconfirmed) close above watched_high
    pending_choch_bear_at: Optional[int] = None  # index of the 1st (unconfirmed) close below watched_low

    for i in range(len(data)):
        # bring any swings confirmed up to this bar into the "watched" levels
        while swing_i < len(swings) and swings[swing_i].index <= i:
            sw = swings[swing_i]
            if sw.kind == "high" and (watched_high is None or sw.price > watched_high.price):
                watched_high = sw
            if sw.kind == "low" and (watched_low is None or sw.price < watched_low.price):
                watched_low = sw
            swing_i += 1

        close = closes[i]

        # --- Bullish break (close above watched_high) ---
        if watched_high is not None and close > watched_high.price:
            if trend == "bullish":
                # BOS -- one body-close beyond the level is enough
                events.append(StructureEvent(index=i, date=data.index[i], price=close, label="BOS_bull"))
                watched_high = None
                pending_choch_bull_at = None
            elif pending_choch_bull_at == i - 1:
                # 2nd CONSECUTIVE body-close beyond the level -- CHOCH confirmed
                events.append(StructureEvent(index=i, date=data.index[i], price=close, label="CHOCH_bull"))
                trend = "bullish"
                watched_high = None
                pending_choch_bull_at = None
            else:
                # 1st close beyond the level -- not confirmed yet, wait for a 2nd
                pending_choch_bull_at = i
        else:
            pending_choch_bull_at = None  # streak broken -- needs to restart from a fresh 1st close

        # --- Bearish break (close below watched_low) ---
        if watched_low is not None and close < watched_low.price:
            if trend == "bearish":
                events.append(StructureEvent(index=i, date=data.index[i], price=close, label="BOS_bear"))
                watched_low = None
                pending_choch_bear_at = None
            elif pending_choch_bear_at == i - 1:
                events.append(StructureEvent(index=i, date=data.index[i], price=close, label="CHOCH_bear"))
                trend = "bearish"
                watched_low = None
                pending_choch_bear_at = None
            else:
                pending_choch_bear_at = i
        else:
            pending_choch_bear_at = None

    return events


def current_bias(events: List[StructureEvent]) -> Bias:
    """The trend implied by the most recent structure event."""
    if not events:
        return "neutral"
    last = events[-1].label
    return "bullish" if "bull" in last else "bearish"


def find_target_level(data: pd.DataFrame, swings: List[SwingPoint],
                       up_to_index: int, direction: str) -> Optional[float]:
    """
    Returns the next unbroken structural level price is heading toward:
    the current unbroken swing high (direction="BUY") or swing low
    (direction="SELL") as of `up_to_index` — i.e. "the next swing
    high/low" a trader would target, not a Fibonacci-derived level.
    """
    watched_high: Optional[SwingPoint] = None
    watched_low: Optional[SwingPoint] = None
    swing_i = 0
    closes = data["Close"].to_numpy()

    for i in range(0, min(up_to_index + 1, len(data))):
        while swing_i < len(swings) and swings[swing_i].index <= i:
            sw = swings[swing_i]
            if sw.kind == "high" and (watched_high is None or sw.price > watched_high.price):
                watched_high = sw
            if sw.kind == "low" and (watched_low is None or sw.price < watched_low.price):
                watched_low = sw
            swing_i += 1

        close = closes[i]
        if watched_high is not None and close > watched_high.price:
            watched_high = None  # broken -- needs a fresh unbroken high
        if watched_low is not None and close < watched_low.price:
            watched_low = None

    if direction == "BUY":
        return watched_high.price if watched_high is not None else None
    return watched_low.price if watched_low is not None else None


def analyze_structure(data: pd.DataFrame, left: int = 2, right: int = 2):
    """Convenience wrapper: returns (swings, events, bias) for a dataframe."""
    swings = find_swings(data, left=left, right=right)
    events = label_structure(data, swings)
    bias = current_bias(events)
    return swings, events, bias
