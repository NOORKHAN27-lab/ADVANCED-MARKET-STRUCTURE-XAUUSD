"""
image_extractor.py
--------------------
Reconstructs approximate OHLC candle data from a chart SCREENSHOT, using
plain color detection -- no AI vision API, no external service, no cost.

How it works
------------
1. The user tells us which RGB color is a bullish (up) candle and which
   is bearish (down), and the price at the very top and very bottom edge
   of the image (read off the chart's own y-axis).
2. We scan the image for pixels matching those colors and group them into
   vertical "blobs" -- one per candle.
3. Within each blob, the wide middle part is the candle body (open/close)
   and the thin center line above/below it is the wick (high/low) -- we
   tell them apart by how many columns of the blob are colored at each
   row (the body spans most of the blob's width, the wick only a sliver).
4. Pixel rows are converted to prices via a straight linear mapping
   between the top/bottom prices you provide.

Limitations (important -- this is a heuristic, not perfect OCR):
  - Works best on a clean, cropped candlestick screenshot with a plain
    background and consistent candle colors (no overlapping indicators/
    text over the candles).
  - Faint/thin wicks on a busy background may be missed.
  - There is no real timestamp info in a screenshot, so the reconstructed
    data uses a synthetic, evenly-spaced index at the timeframe you
    selected -- session-time filtering (NY session) is NOT meaningful
    here and is automatically disabled for image-based analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class ExtractionResult:
    data: pd.DataFrame
    candle_count: int
    warnings: list


def _color_mask(arr: np.ndarray, color: Tuple[int, int, int], tolerance: int) -> np.ndarray:
    diff = np.abs(arr.astype(int) - np.array(color, dtype=int))
    return np.all(diff <= tolerance, axis=-1)


def _find_blobs(mask: np.ndarray, min_gap: int = 1, min_width: int = 1):
    """Group columns with any colored pixel into contiguous candle blobs."""
    col_has_color = mask.any(axis=0)
    blobs = []
    start = None
    gap_run = 0
    for x, has in enumerate(col_has_color):
        if has:
            if start is None:
                start = x
            gap_run = 0
        else:
            if start is not None:
                gap_run += 1
                if gap_run > min_gap:
                    end = x - gap_run
                    if end - start + 1 >= min_width:
                        blobs.append((start, end))
                    start = None
                    gap_run = 0
    if start is not None:
        end = len(col_has_color) - 1
        if end - start + 1 >= min_width:
            blobs.append((start, end))
    return blobs


def _blob_to_ohlc_pixels(mask: np.ndarray, x_start: int, x_end: int, body_ratio: float = 0.55):
    """
    For one candle's column range, split its colored rows into body vs
    wick using how many columns (out of the blob's width) are colored at
    each row. Returns (high_y, low_y, body_top_y, body_bottom_y) in pixel
    row coordinates, or None if nothing usable was found.
    """
    width = x_end - x_start + 1
    sub = mask[:, x_start:x_end + 1]
    coverage = sub.sum(axis=1)  # colored-column count per row
    colored_rows = np.where(coverage > 0)[0]
    if len(colored_rows) == 0:
        return None

    high_y = colored_rows.min()
    low_y = colored_rows.max()

    body_rows = np.where(coverage >= max(1, width * body_ratio))[0]
    if len(body_rows) == 0:
        # no row wide enough to call "body" -- treat the whole thing as a thin doji
        mid = int(np.median(colored_rows))
        return high_y, low_y, mid, mid
    body_top_y = body_rows.min()
    body_bottom_y = body_rows.max()
    return high_y, low_y, body_top_y, body_bottom_y


def extract_candles(
    image: Image.Image,
    bull_color: Tuple[int, int, int],
    bear_color: Tuple[int, int, int],
    top_price: float,
    bottom_price: float,
    color_tolerance: int = 30,
    interval: str = "1h",
) -> ExtractionResult:
    """
    Main entry point: given a PIL image and calibration, returns a
    reconstructed OHLC dataframe (synthetic evenly-spaced datetime index).
    """
    warnings = []
    arr = np.array(image.convert("RGB"))
    height, width, _ = arr.shape

    bull_mask = _color_mask(arr, bull_color, color_tolerance)
    bear_mask = _color_mask(arr, bear_color, color_tolerance)
    combined_mask = bull_mask | bear_mask

    if combined_mask.sum() < 20:
        warnings.append(
            "Very few matching pixels found -- check that the bull/bear colors "
            "match your chart's actual candle colors."
        )

    blobs = _find_blobs(combined_mask, min_gap=1, min_width=1)
    if len(blobs) == 0:
        return ExtractionResult(data=pd.DataFrame(), candle_count=0,
                                 warnings=warnings + ["No candles detected at all."])

    def y_to_price(y: int) -> float:
        frac = y / max(1, height - 1)
        return top_price - frac * (top_price - bottom_price)

    rows = []
    for (x_start, x_end) in blobs:
        result = _blob_to_ohlc_pixels(combined_mask, x_start, x_end)
        if result is None:
            continue
        high_y, low_y, body_top_y, body_bottom_y = result

        blob_bull_votes = bull_mask[:, x_start:x_end + 1].sum()
        blob_bear_votes = bear_mask[:, x_start:x_end + 1].sum()
        is_bull = blob_bull_votes >= blob_bear_votes

        high_price = y_to_price(high_y)
        low_price = y_to_price(low_y)
        body_top_price = y_to_price(body_top_y)
        body_bottom_price = y_to_price(body_bottom_y)

        if is_bull:
            open_price, close_price = body_bottom_price, body_top_price
        else:
            open_price, close_price = body_top_price, body_bottom_price

        rows.append({
            "Open": open_price, "High": high_price, "Low": low_price,
            "Close": close_price, "Volume": 0.0,
        })

    if not rows:
        return ExtractionResult(data=pd.DataFrame(), candle_count=0,
                                 warnings=warnings + ["Candles were located but none could be measured."])

    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(interval, "1h")
    idx = pd.date_range(end=pd.Timestamp.now(), periods=len(rows), freq=freq)

    df = pd.DataFrame(rows, index=idx)

    if len(df) < 20:
        warnings.append(
            f"Only {len(df)} candles detected -- structure/liquidity/zone detection "
            "needs a reasonable number of candles to be meaningful. Try a wider "
            "screenshot with more visible candles."
        )

    return ExtractionResult(data=df, candle_count=len(df), warnings=warnings)


# A few common candle color presets to save the user from having to eyedrop.
COLOR_PRESETS = {
    "TradingView Green/Red": ((8, 153, 129), (242, 54, 69)),
    "Black/White (classic)": ((0, 0, 0), (255, 255, 255)),
    "Blue/Orange": ((41, 98, 255), (255, 152, 0)),
    "Custom (pick below)": None,
}
