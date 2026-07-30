# AI Market Structure Analyzer — XAUUSD & BTCUSD

A multi-timeframe, smart-money-style market structure analyzer built around
one specific strategy sequence — not a generic indicator dump.

**Educational tool. Not financial advice. No trades are placed automatically.**

## Strategy Pipeline

1. **HTF structure (BOS / CHOCH)** — swing highs/lows are detected on the
   higher timeframe, and each break of a swing level is labeled:
   - **BOS** (Break of Structure) — confirmed by **one candle body-close**
     beyond the level (continuation, same direction as the current trend).
   - **CHOCH** (Change of Character) — requires **two consecutive candle
     body-closes** beyond the level (reversal, against the current trend).
     A single close that isn't followed by a second consecutive close does
     NOT confirm the shift — the level stays "watched."
   Both use the candle body close, never the wick.
   The most recent event sets the **HTF bias** (bullish / bearish).

2. **LTF liquidity sweep** — on the lower timeframe, the tool looks for a
   candle that wicks beyond a recent swing high/low (where stop-losses and
   breakout orders cluster) and then **closes back inside the range** — a
   liquidity grab. Only sweeps aligned with the HTF bias are used (e.g. in
   a bullish bias, a sweep of a swing low that then reverses up).

3. **Supply/Demand zone** — right after the sweep, the tool looks for a
   strong displacement move and marks the **last opposite-colored candle**
   before it as the zone (demand zone for bullish setups, supply zone for
   bearish).

4. **Fibonacci — entry confirmation only** — a Fibonacci retracement
   (levels: 0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0 — matching a standard
   TradingView Fib tool) is drawn across the resulting impulse leg. The
   0.618–0.786 band is checked for overlap with the supply/demand zone
   (confluence). **Targets are not Fibonacci-based** — see the next point.

5. **Target = next structural swing level** — the target is the next
   unbroken swing high (for a BUY) or swing low (for a SELL): the next
   liquidity pool / structure level price is likely heading toward, using
   the same swing-detection logic as the BOS/CHOCH engine.

6. **Execution filters** — only setups meeting all of these are kept:
   - **Session** — the sweep must occur within the configured NY session
     window (default 08:00–17:00 America/New_York, DST-aware).
   - **Fresh zone** — the supply/demand zone must not already have been
     used by an earlier setup (no re-testing an already-tapped zone).
   - **Risk:Reward** — a target must land within your configured
     `min_rr`–`max_rr` band (default 1:3 to 1:5); that target becomes the
     trade's actual target.

7. **Backtest evaluation** — once a setup is confirmed (price returns to
   the zone), the tool walks forward through the LTF data to see whether
   the target or the stop-loss was hit first, producing a real **WIN /
   LOSS / OPEN** outcome — and an aggregate win rate — instead of a guess.

8. **Zone touch tagging (A / A+)** — the first time price returns to a
   zone is tagged **A**. If price then leaves the zone, fails to confirm
   a new BOS in the trade's direction, and returns to that SAME zone a
   second time, that second touch is tagged **A+** — a repeat test of a
   level that already held once. The app shows win rate for A and A+
   separately so you can verify whether A+ really does perform better in
   your own data.

## Files

| File | Purpose |
|---|---|
| `data_feed.py` | Fetches XAUUSD (yfinance) and BTCUSD (Binance public API) candles. Falls back to synthetic data offline. |
| `image_extractor.py` | Reconstructs OHLC candles from an uploaded chart **screenshot** using color-based detection — no AI/API needed. |
| `structure.py` | Fractal swing detection + BOS/CHOCH structure labeling. |
| `liquidity.py` | Liquidity sweep detection. |
| `zones.py` | Supply/demand zone detection + Fibonacci level calculation. |
| `filters.py` | Session window, zone-freshness, and Risk:Reward filters. |
| `backtest.py` | Simulates each setup forward to determine WIN/LOSS/OPEN. |
| `signal_engine.py` | Combines everything above into full trade `Setup` objects, filtered and backtested. |
| `chart.py` | Candlestick chart rendering with structure/zone/fib overlays (pure matplotlib). |
| `app.py` | Streamlit dashboard — the full tool. |
| `requirements.txt` | Dependencies. |

## Two ways to feed it data

**1. Live Market Data** — fetches real XAUUSD/BTCUSD candles automatically.

**2. Upload Chart Image** — upload a screenshot of a chart (TradingView, your
broker, anywhere) instead. You provide the candle colors, the price shown
at the top/bottom edge of the image (read off the chart's own y-axis),
and the timeframe. The tool detects each candle by color, splits its body
from its wick, converts pixels to prices, and runs the exact same
BOS/CHOCH → sweep → zone → fib → A/A+ pipeline on the reconstructed data.
A preview of the reconstructed candles is shown so you can sanity-check
the extraction. Session-time filtering is disabled in this mode since a
screenshot has no real timestamps.

**Image mode limitations:** works best on a clean, cropped candlestick
screenshot with a plain background and no overlapping indicators/text on
top of the candles, with plenty of visible candles.

## Run locally

```bash
pip install streamlit yfinance requests pandas numpy matplotlib
streamlit run app.py
```

Pick a symbol (XAUUSD or BTCUSD), an HTF (bias) and LTF (entry) timeframe,
and swing sensitivity, then **Run Analysis**. You'll get:
- A bias card (current HTF trend)
- An HTF chart with every BOS/CHOCH label
- An LTF chart showing the latest sweep, zone, and Fibonacci levels
- A list of every setup found, with entry zone, stop-loss, targets, and
  the exact confluence reasoning behind each one

## Deploy publicly (free)

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **Create app** → select this repo → branch `main` → main file `app.py` → **Deploy**.

## Notes & limitations

- Swing/structure detection is a standard non-repainting fractal method —
  it will differ somewhat from manual chart reading, especially right at
  the edge of the visible data (the most recent 1-2 swings need a few bars
  to confirm).
- "Supply/demand zone" and "liquidity sweep" have several valid definitions
  across different trading styles — this implementation follows the
  sequence described above; thresholds (`displacement_factor`,
  `tolerance_pct`, lookahead windows) are adjustable in `zones.py` /
  `liquidity.py` if your own rules are stricter or looser.
- This surfaces trade ideas for your own evaluation — always confirm on
  your own charts before acting on anything shown here.
- **Win rate needs a real sample.** With the session + fresh-zone + tight
  R:R filters all on, very few setups will qualify in a short LTF window —
  that's by design (it's meant to be selective), but it also means the
  displayed win rate isn't statistically meaningful until you run it over
  a much longer history (thousands of LTF bars) so there are enough closed
  trades to judge. The app warns you when the closed-trade count is small.
- If your live win rate is meaningfully different from what this tool
  backtests, the gap is almost always in how the automated detection
  approximates a specific rule — e.g. your definition of "fresh zone",
  "displacement", or exact NY session hours may be stricter/looser than
  the defaults here. All of these are adjustable (`filters.py`,
  `zones.py`, sidebar controls).

---
Built by Noor Ahmed Khan
