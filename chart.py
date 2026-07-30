"""
chart.py
--------
Candlestick chart rendering with structure/zone/fibonacci overlays,
using plain matplotlib (no extra plotting dependency required).
"""

from typing import List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from structure import StructureEvent
from liquidity import LiquiditySweep
from zones import Zone

BULL_COLOR = "#3FD08C"
BEAR_COLOR = "#E0576B"
WICK_COLOR = "#8A8FA3"


def plot_candles(ax, data: pd.DataFrame, width_frac: float = 0.6):
    """Draw OHLC candlesticks onto a matplotlib axis."""
    x = mdates.date2num(data.index.to_pydatetime())
    if len(x) > 1:
        bar_width = (x[1] - x[0]) * width_frac
    else:
        bar_width = 0.5

    opens = data["Open"].to_numpy()
    closes = data["Close"].to_numpy()
    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()

    for xi, o, c, h, l in zip(x, opens, closes, highs, lows):
        color = BULL_COLOR if c >= o else BEAR_COLOR
        ax.plot([xi, xi], [l, h], color=WICK_COLOR, linewidth=0.7, zorder=2)
        rect = Rectangle(
            (xi - bar_width / 2, min(o, c)), bar_width, max(abs(c - o), 1e-6),
            facecolor=color, edgecolor=color, zorder=3,
        )
        ax.add_patch(rect)

    ax.xaxis_date()
    ax.set_xlim(x.min() - bar_width * 2, x.max() + bar_width * 2)


def style_axis(ax, title: str):
    ax.set_title(title, color="#EDEBF7", fontsize=12)
    ax.tick_params(colors="#8A8FA3", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2A2740")
    ax.grid(alpha=0.12, color="#8A8FA3")
    ax.set_facecolor("#14121F")


def plot_winrate_gauge(win_rate: float, wins: int, losses: int):
    """Donut-style gauge showing the win rate as a premium visual centerpiece."""
    fig, ax = plt.subplots(figsize=(3.6, 3.6), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    if win_rate >= 55:
        ring_color = "#3FD08C"
    elif win_rate >= 35:
        ring_color = "#C7A15A"
    else:
        ring_color = "#E0576B"

    remainder = max(0.0, 100 - win_rate)
    wedges, _ = ax.pie(
        [win_rate, remainder] if (win_rate + remainder) > 0 else [1],
        startangle=90, counterclock=False,
        colors=[ring_color, "#231F38"] if (win_rate + remainder) > 0 else ["#231F38"],
        wedgeprops=dict(width=0.28, edgecolor="#0B0A14", linewidth=2),
    )
    ax.text(0, 0.12, f"{win_rate:.0f}%", ha="center", va="center",
            fontsize=30, fontweight="bold", color="#EDEBF7", family="monospace")
    ax.text(0, -0.28, "WIN RATE", ha="center", va="center",
            fontsize=9, color="#8A8FA3", family="monospace")
    ax.text(0, -1.35, f"{wins}W · {losses}L", ha="center", va="center",
            fontsize=9.5, color=ring_color, family="monospace")
    return fig


def plot_htf_structure(fig_ax, data: pd.DataFrame, events: List[StructureEvent]):
    fig, ax = fig_ax
    plot_candles(ax, data)
    for e in events:
        color = BULL_COLOR if "bull" in e.label else BEAR_COLOR
        marker = "^" if "bull" in e.label else "v"
        ax.scatter(mdates.date2num(pd.Timestamp(e.date).to_pydatetime()), e.price,
                   color=color, marker=marker, s=70, zorder=5)
        short = "CHOCH" if "CHOCH" in e.label else "BOS"
        ax.annotate(short, (mdates.date2num(pd.Timestamp(e.date).to_pydatetime()), e.price),
                   textcoords="offset points", xytext=(0, 10 if "bull" in e.label else -14),
                   ha="center", fontsize=7, color=color)
    style_axis(ax, "HTF — Market Structure (BOS / CHOCH)")


def plot_ltf_setup(fig_ax, data: pd.DataFrame, sweep: Optional[LiquiditySweep],
                    zone: Optional[Zone], fib_levels: Optional[dict]):
    fig, ax = fig_ax
    plot_candles(ax, data)

    if sweep is not None:
        xi = mdates.date2num(pd.Timestamp(sweep.date).to_pydatetime())
        color = BULL_COLOR if sweep.direction == "bullish" else BEAR_COLOR
        ax.scatter(xi, sweep.wick_extreme, color=color, marker="x", s=90, zorder=6)
        ax.annotate("SWEEP", (xi, sweep.wick_extreme), textcoords="offset points",
                   xytext=(6, 0), fontsize=7, color=color)

    if zone is not None:
        x = mdates.date2num(data.index.to_pydatetime())
        zone_color = "#C7A15A" if zone.kind == "demand" else "#8A6BC7"
        ax.axhspan(zone.bottom, zone.top, xmin=0, xmax=1, color=zone_color, alpha=0.15, zorder=1)
        ax.text(x[-1], (zone.top + zone.bottom) / 2, f"  {zone.kind}", color=zone_color,
               fontsize=8, va="center")

    if fib_levels:
        x = mdates.date2num(data.index.to_pydatetime())
        for label, price in fib_levels.items():
            is_key = label in ("0.618", "0.786")
            ax.axhline(price, color="#9C8FF0" if is_key else "#4A4568",
                      linewidth=1.1 if is_key else 0.6, linestyle="--", alpha=0.8, zorder=1)
            ax.text(x[0], price, f" {label}", color="#9C8FF0" if is_key else "#6C6690",
                   fontsize=6.5, va="center")

    style_axis(ax, "LTF — Liquidity Sweep + Zone + Fibonacci")
