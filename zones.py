"""
zones.py
--------
Two pieces of confluence used after a liquidity sweep:

1. Supply/Demand zone — the last opposing candle right before an
   impulsive ("displacement") move. For a bullish reversal, the demand
   zone is the last bearish (red) candle before a strong up-leg; for a
   bearish reversal, the supply zone is the last bullish (green) candle
   before a strong down-leg.

2. Fibonacci retracement/extension — drawn across the impulse leg that
   follows a sweep, giving a confirmation zone (0.618-0.79 retracement)
   and price targets (extensions).
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from liquidity import LiquiditySweep


@dataclass
class Zone:
    start_index: int
    end_index: int
    top: float
    bottom: float
    kind: Literal["demand", "supply"]


def find_zone_after_sweep(data: pd.DataFrame, sweep: LiquiditySweep,
                           lookahead: int = 12, displacement_factor: float = 1.6) -> Optional[Zone]:
    """
    Looks forward from the sweep bar for a displacement move (a candle or
    small run of candles whose body is much larger than the recent
    average), then returns the last opposite-colored candle immediately
    before that impulse as the supply/demand zone.
    """
    opens = data["Open"].to_numpy()
    closes = data["Close"].to_numpy()
    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()

    bodies = np.abs(closes - opens)
    avg_body = np.mean(bodies[max(0, sweep.index - 20):sweep.index + 1]) or 1e-9

    end = min(sweep.index + lookahead, len(data) - 1)
    want_bull_impulse = sweep.direction == "bullish"

    for i in range(sweep.index, end):
        body = closes[i] - opens[i]
        is_impulse = abs(body) > avg_body * displacement_factor
        right_direction = (body > 0) if want_bull_impulse else (body < 0)

        if is_impulse and right_direction:
            # walk backward to find the last opposite-colored candle before this impulse
            j = i - 1
            while j > sweep.index - 3 and j >= 0:
                candle_bull = closes[j] > opens[j]
                if want_bull_impulse and not candle_bull:
                    return Zone(start_index=j, end_index=j, top=highs[j], bottom=lows[j], kind="demand")
                if not want_bull_impulse and candle_bull:
                    return Zone(start_index=j, end_index=j, top=highs[j], bottom=lows[j], kind="supply")
                j -= 1
            # fallback: use the sweep candle itself as the zone
            kind = "demand" if want_bull_impulse else "supply"
            return Zone(start_index=sweep.index, end_index=sweep.index,
                        top=highs[sweep.index], bottom=lows[sweep.index], kind=kind)

    return None


def compute_fibonacci(low_price: float, high_price: float, direction: str) -> dict:
    """
    Standard retracement levels across one impulse leg, matching the exact
    levels enabled in the user's Fibonacci tool: 0, 0.236, 0.382, 0.5,
    0.618, 0.786, 1.0. Used only for the entry confirmation zone
    (0.618-0.786) -- NOT for targets (targets come from the next
    swing high/low instead, see structure.find_target_level).

    `direction="bullish"` means the leg ran low -> high (retracements
    measured back down from the high); `"bearish"` means high -> low.
    """
    diff = high_price - low_price
    if direction == "bullish":
        levels = {
            "0.0 (high)": high_price,
            "0.236": high_price - 0.236 * diff,
            "0.382": high_price - 0.382 * diff,
            "0.5": high_price - 0.5 * diff,
            "0.618": high_price - 0.618 * diff,
            "0.786": high_price - 0.786 * diff,
            "1.0 (low)": low_price,
        }
    else:
        levels = {
            "0.0 (low)": low_price,
            "0.236": low_price + 0.236 * diff,
            "0.382": low_price + 0.382 * diff,
            "0.5": low_price + 0.5 * diff,
            "0.618": low_price + 0.618 * diff,
            "0.786": low_price + 0.786 * diff,
            "1.0 (high)": high_price,
        }
    return levels
