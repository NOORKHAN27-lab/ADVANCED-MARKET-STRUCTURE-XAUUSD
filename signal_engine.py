"""
signal_engine.py
-----------------
Combines all confluence pieces into one trade setup, following the exact
strategy sequence, then applies your execution rules and measures a real
outcome (not a guess):

    1. HTF market structure (BOS/CHOCH) -> establishes bias (bullish/bearish)
    2. LTF liquidity sweep in the direction of that bias
    3. Supply/Demand zone formed right after the sweep -- must be FRESH
       (first time price returns to it; a zone already used by an earlier
       setup is skipped)
    4. Fibonacci retracement of the resulting impulse leg -> ENTRY
       CONFIRMATION only (0.618-0.786 zone must overlap the supply/demand
       zone). Targets are NOT Fibonacci-based -- see step 5.
    5. Target = the next unbroken structural swing level (the next swing
       high above price for a BUY, or swing low below price for a SELL) --
       i.e. the next liquidity pool / structure level in the direction of
       the trade.
    6. Session filter -- only setups whose sweep occurs in the configured
       session window (default: New York session)
    7. Risk:Reward filter -- only keep a setup if that structural target
       lands within [min_rr, max_rr] (default 1:3 to 1:5)
    8. Zone touch tagging (A / A+) -- the first time price returns to the
       zone is tagged "A". If price then leaves, fails to confirm a new
       BOS in the trade's direction, and comes back to the SAME zone a
       second time, that second touch is tagged "A+" -- a higher-quality
       repeat test of the same level (per your own stats: A ~65% win rate,
       A+ ~85%). Both are surfaced as separate setups so you can compare.
    9. Backtest evaluation -- walk forward through the LTF data to see
       whether the target or stop was hit first -> WIN / LOSS / OPEN

This module produces `Setup` objects -- informational trade ideas for you
to evaluate, not auto-executed trades. Nothing here places real orders or
constitutes financial advice.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import pandas as pd

from structure import analyze_structure, find_target_level, find_target_candidates
from liquidity import LiquiditySweep, find_liquidity_sweeps
from zones import Zone, find_zone_after_sweep, compute_fibonacci
from filters import in_ny_session, zones_overlap, select_target_by_rr, find_valid_retracement_entry
from backtest import simulate_outcome


@dataclass
class Setup:
    direction: Literal["BUY", "SELL"]
    sweep: LiquiditySweep
    zone: Zone
    fib_levels: dict           # confirmation-zone reference only (0.618/0.786 etc.)
    entry_zone: tuple          # (low, high)
    stop_loss: float
    chosen_target: float       # next swing high/low (structural target)
    risk_reward: float
    touch_label: str = "A"     # "A" (first touch) or "A+" (confirmed 2nd touch)
    confluence_notes: List[str] = field(default_factory=list)
    confirmed: bool = False    # True if price has returned into the zone
    outcome: str = "OPEN"      # "WIN" | "LOSS" | "OPEN"
    exit_price: Optional[float] = None
    bars_held: Optional[int] = None


def build_setups(ltf_data: pd.DataFrame, htf_bias: str,
                  ltf_left: int = 2, ltf_right: int = 2,
                  session_only: bool = True, session_start_hour: int = 8, session_end_hour: int = 17,
                  fresh_zones_only: bool = True,
                  min_rr: float = 3.0, max_rr: float = 5.0,
                  require_fib_confluence: bool = False,
                  htf_zones: Optional[List[tuple]] = None,
                  require_htf_confluence: bool = False,
                  evaluate: bool = True) -> List[Setup]:
    """
    Given LTF candle data and the HTF bias, find liquidity sweeps aligned
    with that bias and build full Setup objects -- filtered down to only
    the ones matching session, freshness, and risk:reward rules, then
    (optionally) backtested for a real WIN/LOSS/OPEN outcome. Each zone
    can produce up to two setups: the first touch ("A") and, if price
    fails to confirm a new BOS before retesting the same zone, a second
    ("A+") setup.

    Two additional optional quality filters (both off by default):
      - `require_fib_confluence`: only keep a setup if its zone overlaps
        the 0.618-0.786 Fibonacci confirmation band.
      - `require_htf_confluence`: only keep a setup if its zone overlaps
        one of the given `htf_zones` (higher-timeframe supply/demand
        zones) -- multi-timeframe confluence.
    """
    if htf_bias not in ("bullish", "bearish"):
        return []

    ltf_swings, ltf_events, _ = analyze_structure(ltf_data, left=ltf_left, right=ltf_right)
    sweeps = find_liquidity_sweeps(ltf_data, ltf_swings)

    wanted_direction = "bullish" if htf_bias == "bullish" else "bearish"
    aligned_sweeps = [s for s in sweeps if s.direction == wanted_direction]

    setups: List[Setup] = []
    used_zones: List[tuple] = []  # (bottom, top) of zones already claimed by an earlier sweep
    highs = ltf_data["High"].to_numpy()
    lows = ltf_data["Low"].to_numpy()
    closes = ltf_data["Close"].to_numpy()
    htf_zones = htf_zones or []

    for sweep in aligned_sweeps:
        if session_only and not in_ny_session(sweep.date, session_start_hour, session_end_hour):
            continue

        zone = find_zone_after_sweep(ltf_data, sweep)
        if zone is None:
            continue

        zone_range = (zone.bottom, zone.top)

        if fresh_zones_only and any(zones_overlap(zone_range, used) for used in used_zones):
            continue

        if require_htf_confluence:
            if not any(zones_overlap(zone_range, htf_z) for htf_z in htf_zones):
                continue

        search_end = min(zone.end_index + 30, len(ltf_data) - 1)
        entry_low, entry_high = zone.bottom, zone.top

        if sweep.direction == "bullish":
            leg_high = highs[zone.end_index:search_end + 1].max()
            fib = compute_fibonacci(low_price=sweep.wick_extreme, high_price=leg_high, direction="bullish")
            stop_loss = sweep.wick_extreme * 0.999
            direction = "BUY"
        else:
            leg_low = lows[zone.end_index:search_end + 1].min()
            fib = compute_fibonacci(low_price=leg_low, high_price=sweep.wick_extreme, direction="bearish")
            stop_loss = sweep.wick_extreme * 1.001
            direction = "SELL"

        # --- Target = next unbroken swing high/low (structural level), NOT Fibonacci ---
        # Consider ALL still-unbroken levels ahead (nearest first), not just the very
        # next one -- so a level further out can be used if it's the one that actually
        # fits the desired Risk:Reward band.
        target_candidates = find_target_candidates(ltf_data, ltf_swings, up_to_index=zone.end_index, direction=direction)
        if not target_candidates:
            continue

        rr_result = select_target_by_rr((entry_low, entry_high), stop_loss, target_candidates, direction, min_rr, max_rr)
        if rr_result is None:
            continue
        chosen_target, rr = rr_result

        fib_zone_low, fib_zone_high = sorted([fib["0.618"], fib["0.786"]])
        overlaps_fib = not (entry_high < fib_zone_low or entry_low > fib_zone_high)

        if require_fib_confluence and not overlaps_fib:
            continue

        base_notes = [
            f"Liquidity swept at {sweep.swept_level:.2f} (wick to {sweep.wick_extreme:.2f}), "
            f"aligned with {htf_bias} HTF bias.",
            f"{zone.kind.capitalize()} zone found at {zone.bottom:.2f}-{zone.top:.2f} (fresh).",
            f"Target = next {'swing high' if direction == 'BUY' else 'swing low'} at {chosen_target:.2f} "
            f"(structural level, not a Fibonacci extension).",
            f"Risk:Reward = 1:{rr:.2f} -- within the {min_rr:.0f}-{max_rr:.0f} target range.",
        ]
        if overlaps_fib:
            base_notes.append("Zone overlaps the 0.618-0.786 Fibonacci confirmation band -- confluence.")
        if session_only:
            base_notes.append("Sweep occurred within the configured NY session window.")
        if require_htf_confluence:
            base_notes.append("Zone also overlaps a higher-timeframe supply/demand zone -- multi-timeframe confluence.")

        # --- First touch (A): require a genuine, progressive pullback into
        # the zone -- at least 2 consecutive counter-trend candles, each
        # closing further than the last -- not just any single touch. ---
        entry_index = find_valid_retracement_entry(ltf_data, (entry_low, entry_high),
                                                     start_index=zone.end_index, direction=direction)
        confirmed = entry_index is not None
        if confirmed:
            base_notes.append(
                f"Valid {'red' if direction == 'BUY' else 'green'}-candle retracement into the zone "
                "(2+ consecutive candles, each closing further) -- not just a wick touch."
            )

        setup_a = Setup(
            direction=direction, sweep=sweep, zone=zone, fib_levels=fib,
            entry_zone=(entry_low, entry_high), stop_loss=stop_loss,
            chosen_target=chosen_target, risk_reward=rr, touch_label="A",
            confluence_notes=list(base_notes), confirmed=confirmed,
        )
        if evaluate and confirmed:
            result = simulate_outcome(ltf_data, entry_index, (entry_low, entry_high),
                                       stop_loss, chosen_target, direction)
            setup_a.outcome = result.outcome
            setup_a.exit_price = result.exit_price
            setup_a.bars_held = result.bars_held
        setups.append(setup_a)

        # --- Second touch (A+) -- only if price left the zone, failed to confirm a
        # new BOS in this trade's direction, and comes back with another valid
        # progressive retracement into the SAME zone ---
        if confirmed:
            left_index = None
            for i in range(entry_index + 1, min(entry_index + 150, len(closes))):
                if not (entry_low <= closes[i] <= entry_high):
                    left_index = i
                    break

            second_index = None
            if left_index is not None:
                second_index = find_valid_retracement_entry(
                    ltf_data, (entry_low, entry_high), start_index=left_index,
                    direction=direction, max_lookahead=150,
                )

            if second_index is not None:
                bos_label = "BOS_bull" if direction == "BUY" else "BOS_bear"
                bos_between = any(
                    e.label == bos_label and entry_index < e.index < second_index
                    for e in ltf_events
                )
                if not bos_between:
                    a_plus_notes = list(base_notes) + [
                        "Second touch of the same zone (A+) -- price left the zone but did NOT "
                        "confirm a new BOS in this direction, then gave another valid progressive "
                        "retracement back into the same zone.",
                    ]
                    setup_a_plus = Setup(
                        direction=direction, sweep=sweep, zone=zone, fib_levels=fib,
                        entry_zone=(entry_low, entry_high), stop_loss=stop_loss,
                        chosen_target=chosen_target, risk_reward=rr, touch_label="A+",
                        confluence_notes=a_plus_notes, confirmed=True,
                    )
                    if evaluate:
                        result2 = simulate_outcome(ltf_data, second_index, (entry_low, entry_high),
                                                    stop_loss, chosen_target, direction)
                        setup_a_plus.outcome = result2.outcome
                        setup_a_plus.exit_price = result2.exit_price
                        setup_a_plus.bars_held = result2.bars_held
                    setups.append(setup_a_plus)

        used_zones.append(zone_range)

    return setups


def win_rate_stats(setups: List[Setup]) -> dict:
    """Aggregate WIN/LOSS/OPEN counts and win rate across a list of Setups."""
    closed = [s for s in setups if s.outcome in ("WIN", "LOSS")]
    wins = sum(1 for s in closed if s.outcome == "WIN")
    losses = sum(1 for s in closed if s.outcome == "LOSS")
    open_count = sum(1 for s in setups if s.outcome == "OPEN")
    win_rate = (wins / len(closed) * 100) if closed else 0.0
    return {
        "total_setups": len(setups),
        "wins": wins,
        "losses": losses,
        "open": open_count,
        "closed": len(closed),
        "win_rate": win_rate,
    }


def win_rate_by_touch(setups: List[Setup]) -> dict:
    """Split win-rate stats by touch_label ('A' vs 'A+') for direct comparison."""
    result = {}
    for label in ("A", "A+"):
        subset = [s for s in setups if s.touch_label == label]
        result[label] = win_rate_stats(subset)
    return result


def expectancy_stats(setups: List[Setup]) -> dict:
    """
    R-multiple expectancy and profit factor across closed setups -- the
    fuller picture beyond win rate alone. Each WIN scores +risk_reward R
    (the target was hit), each LOSS scores -1R (full risk lost, since
    stop-loss is what defines 1R). A strategy can have a low win rate and
    still be strongly profitable if the average win is large relative to
    the average loss (which is exactly what a wide Risk:Reward band like
    1:3-1:5 is designed to produce).
    """
    closed = [s for s in setups if s.outcome in ("WIN", "LOSS")]
    if not closed:
        return {"expectancy_r": 0.0, "profit_factor": 0.0, "total_r": 0.0,
                "avg_win_r": 0.0, "avg_loss_r": -1.0, "closed": 0}

    r_values = [s.risk_reward if s.outcome == "WIN" else -1.0 for s in closed]
    wins_r = [r for r in r_values if r > 0]
    losses_r = [r for r in r_values if r < 0]

    total_r = sum(r_values)
    expectancy = total_r / len(closed)
    gross_win = sum(wins_r) if wins_r else 0.0
    gross_loss = abs(sum(losses_r)) if losses_r else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    return {
        "expectancy_r": expectancy,
        "profit_factor": profit_factor,
        "total_r": total_r,
        "avg_win_r": (sum(wins_r) / len(wins_r)) if wins_r else 0.0,
        "avg_loss_r": (sum(losses_r) / len(losses_r)) if losses_r else -1.0,
        "closed": len(closed),
    }
