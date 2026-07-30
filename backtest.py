"""
backtest.py
-----------
Simulates each Setup forward through the LTF candles to determine whether
its target or stop-loss was hit first — turning "here's a setup" into an
actual measurable WIN / LOSS / OPEN outcome and win rate, instead of a
manual eyeball check.

Rule: starting from the bar where price first entered the entry zone,
scan forward bar-by-bar. Whichever level (target or stop) is touched
first decides the outcome. If neither is hit before the data runs out,
the setup is marked OPEN (still active / inconclusive).
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd

Outcome = Literal["WIN", "LOSS", "OPEN"]


@dataclass
class BacktestOutcome:
    outcome: Outcome
    exit_index: Optional[int]
    exit_price: Optional[float]
    bars_held: Optional[int]


def find_entry_index(data: pd.DataFrame, entry_zone, start_index: int, lookahead: int = 60) -> Optional[int]:
    """First bar at/after start_index whose Close falls inside the entry zone."""
    low, high = entry_zone
    closes = data["Close"].to_numpy()
    end = min(start_index + lookahead, len(data))
    for i in range(start_index, end):
        if low <= closes[i] <= high:
            return i
    return None


def simulate_outcome(data: pd.DataFrame, entry_index: int, entry_zone,
                      stop_loss: float, target: float,
                      direction: Literal["BUY", "SELL"]) -> BacktestOutcome:
    """Walk forward from entry_index until target or stop is hit."""
    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()

    for i in range(entry_index, len(data)):
        if direction == "BUY":
            hit_stop = lows[i] <= stop_loss
            hit_target = highs[i] >= target
        else:
            hit_stop = highs[i] >= stop_loss
            hit_target = lows[i] <= target

        if hit_stop:
            return BacktestOutcome("LOSS", i, stop_loss, i - entry_index)
        if hit_target:
            return BacktestOutcome("WIN", i, target, i - entry_index)

    return BacktestOutcome("OPEN", None, None, None)


def summarize_outcomes(outcomes: List[BacktestOutcome]) -> dict:
    closed = [o for o in outcomes if o.outcome in ("WIN", "LOSS")]
    wins = sum(1 for o in closed if o.outcome == "WIN")
    losses = sum(1 for o in closed if o.outcome == "LOSS")
    still_open = sum(1 for o in outcomes if o.outcome == "OPEN")
    total_closed = len(closed)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    return {
        "total_setups": len(outcomes),
        "wins": wins,
        "losses": losses,
        "open": still_open,
        "closed": total_closed,
        "win_rate": win_rate,
    }
