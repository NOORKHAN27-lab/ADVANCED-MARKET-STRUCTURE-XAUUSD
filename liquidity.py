"""
liquidity.py
------------
Detects "liquidity sweeps" (a.k.a. stop hunts / liquidity grabs): price
pierces beyond a recent swing high/low — where resting stop-loss and
breakout orders cluster — then quickly rejects back inside the range.

A bullish liquidity sweep (sell-side liquidity grab) = candle wicks
below a recent swing low but closes back above it → often precedes a
bullish reversal, so we look for these when the HTF bias is bullish.

A bearish liquidity sweep (buy-side liquidity grab) = candle wicks
above a recent swing high but closes back below it → looked for when
HTF bias is bearish.
"""

from dataclasses import dataclass
from typing import List, Literal

import pandas as pd

from structure import SwingPoint


@dataclass
class LiquiditySweep:
    index: int
    date: object
    swept_level: float
    wick_extreme: float
    close: float
    direction: Literal["bullish", "bearish"]  # reversal direction the sweep favors


def find_liquidity_sweeps(data: pd.DataFrame, swings: List[SwingPoint],
                           tolerance_pct: float = 0.0015) -> List[LiquiditySweep]:
    """
    For each swing low, scan forward for a later candle whose Low pierces
    below it (within `tolerance_pct` beyond, to allow for near-equal-lows
    sweeps) but whose Close rejects back above → bullish sweep. Mirror
    logic for swing highs → bearish sweep. Only the first qualifying
    sweep after each swing is kept.
    """
    sweeps: List[LiquiditySweep] = []
    lows = data["Low"].to_numpy()
    highs = data["High"].to_numpy()
    closes = data["Close"].to_numpy()

    swing_lows = [s for s in swings if s.kind == "low"]
    swing_highs = [s for s in swings if s.kind == "high"]

    for sw in swing_lows:
        level = sw.price
        for j in range(sw.index + 1, len(data)):
            if lows[j] < level and closes[j] > level:
                sweeps.append(LiquiditySweep(
                    index=j, date=data.index[j], swept_level=level,
                    wick_extreme=lows[j], close=closes[j], direction="bullish",
                ))
                break
            if closes[j] < level * (1 - tolerance_pct):
                break  # price broke down decisively — no sweep, real breakdown

    for sw in swing_highs:
        level = sw.price
        for j in range(sw.index + 1, len(data)):
            if highs[j] > level and closes[j] < level:
                sweeps.append(LiquiditySweep(
                    index=j, date=data.index[j], swept_level=level,
                    wick_extreme=highs[j], close=closes[j], direction="bearish",
                ))
                break
            if closes[j] > level * (1 + tolerance_pct):
                break

    sweeps.sort(key=lambda s: s.index)
    return sweeps
