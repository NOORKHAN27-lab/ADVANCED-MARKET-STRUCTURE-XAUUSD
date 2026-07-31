"""
image_extractor.py
--------------------
Reconstructs approximate OHLC candle data from a chart SCREENSHOT, using
plain color detection -- no AI vision API, no external service, no cost.
The price axis (top/bottom price) is read automatically via OCR on the
chart's own y-axis labels, so no manual price entry is needed.

How it works
------------
1. We OCR the right-hand price-axis labels (e.g. "4,170.000", "4,160.000"
   ...) and their pixel positions, then fit a straight line to convert
   any pixel row to a price -- no manual calibration needed.
2. You tell us which RGB color is a bullish (up) candle and which is
   bearish (down) (a few presets are provided).
3. We scan the image for pixels matching those colors and group them into
   vertical "blobs" -- one per candle. Oversized blobs (price badges,
   buttons -- not real candles) are filtered out by width.
4. Within each blob, the wide middle part is the candle body (open/close)
   and the thin center line above/below it is the wick (high/low) -- we
   tell them apart by how many columns of the blob are colored at each
   row (the body spans most of the blob's width, the wick only a sliver).
5. Pixel rows are converted to prices via the OCR-fitted axis mapping.

Limitations (important -- this is a heuristic, not perfect OCR):
  - Works best on a clean candlestick screenshot with visible y-axis price
    labels and consistent candle colors (no overlapping indicators/text
    directly over the candles).
  - If OCR can't find at least 2 clear price labels, you'll need to enter
    the top/bottom price manually as a fallback.
  - There is no real timestamp info in a screenshot, so the reconstructed
    data uses a synthetic, evenly-spaced index at the timeframe you
    selected -- session-time filtering (NY session) is NOT meaningful
    here and is automatically disabled for image-based analysis.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


_PRICE_TOKEN_RE = re.compile(r"^\d{1,3}(,\d{3})*(\.\d+)?$")


def auto_detect_price_axis(
    image: Image.Image,
    axis_strip_frac: float = 0.15,
    upscale: int = 3,
    min_labels: int = 2,
) -> Optional[Tuple[float, float]]:
    """
    Reads the chart's own right-hand y-axis price labels via OCR and fits
    a straight line to convert pixel rows to prices. Returns
    (top_price, bottom_price) for the full image height, or None if not
    enough labels could be confidently read (caller should fall back to
    manual entry in that case).
    """
    if not OCR_AVAILABLE:
        return None

    rgb = image.convert("RGB")
    width, height = rgb.size
    strip = rgb.crop((int(width * (1 - axis_strip_frac)), 0, width, height))
    strip_big = strip.resize((strip.width * upscale, strip.height * upscale), Image.LANCZOS)

    try:
        data = pytesseract.image_to_data(strip_big, config="--psm 6", output_type=Output.DICT)
    except Exception:
        return None

    points = []
    for i, text in enumerate(data["text"]):
        token = text.strip()
        if not token or not _PRICE_TOKEN_RE.match(token):
            continue
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0
        if conf < 40:
            continue
        price = float(token.replace(",", ""))
        y_center = (data["top"][i] + data["height"][i] / 2) / upscale
        points.append((y_center, price))

    if len(points) < min_labels:
        return None

    ys = np.array([p[0] for p in points])
    prices = np.array([p[1] for p in points])

    # Sanity check: axis prices should be monotonically decreasing as y
    # increases (top of image = highest price). Reject if that's not
    # roughly true -- likely OCR misreads.
    order = np.argsort(ys)
    if not np.all(np.diff(prices[order]) <= 0):
        # allow a little noise but bail if it's clearly not monotonic
        sign_changes = np.sum(np.diff(prices[order]) > 0)
        if sign_changes > max(1, len(points) // 4):
            return None

    a, b = np.polyfit(ys, prices, 1)
    top_price = b
    bottom_price = a * (height - 1) + b
    return float(top_price), float(bottom_price)


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
    top_price: Optional[float] = None,
    bottom_price: Optional[float] = None,
    color_tolerance: int = 30,
    interval: str = "1h",
) -> ExtractionResult:
    """
    Main entry point: given a PIL image, returns a reconstructed OHLC
    dataframe (synthetic evenly-spaced datetime index). If `top_price`/
    `bottom_price` aren't given, they're auto-detected via OCR on the
    chart's own y-axis labels -- pass them explicitly only if OCR fails
    or you want to override it.
    """
    warnings = []

    if top_price is None or bottom_price is None:
        detected = auto_detect_price_axis(image)
        if detected is None:
            return ExtractionResult(
                data=pd.DataFrame(), candle_count=0,
                warnings=["Could not automatically read the price axis from this image. "
                          "Please enter the top/bottom price manually."],
            )
        top_price, bottom_price = detected
        warnings.append(f"Auto-detected price axis: {top_price:.2f} (top) to {bottom_price:.2f} (bottom).")

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

    blobs = _find_blobs(combined_mask, min_gap=0, min_width=1)
    if len(blobs) == 0:
        return ExtractionResult(data=pd.DataFrame(), candle_count=0,
                                 warnings=warnings + ["No candles detected at all."])

    # Filter out non-candle UI elements (price badges, buttons) that got
    # picked up by the color match -- real candles cluster tightly around
    # one typical width; a badge/box is a clear width outlier.
    widths_arr = np.array([e - s + 1 for s, e in blobs])
    median_width = float(np.median(widths_arr))
    width_cap = max(6.0, median_width * 8)

    # Also reject sparse/low-density blobs -- a real candle (body+wick) is
    # a solidly-filled shape, while gridlines, dotted reference lines, or
    # anti-aliased UI edges that happen to match the color tolerance tend
    # to be sparse/broken within their own bounding box.
    kept_blobs = []
    dropped_sparse = 0
    for (s, e), w in zip(blobs, widths_arr):
        if w > width_cap:
            continue
        sub = combined_mask[:, s:e + 1]
        rows_with_color = np.where(sub.any(axis=1))[0]
        if len(rows_with_color) == 0:
            continue
        bbox_h = rows_with_color.max() - rows_with_color.min() + 1
        density = sub.sum() / (w * bbox_h)
        if density < 0.15:
            dropped_sparse += 1
            continue
        kept_blobs.append((s, e))

    if len(blobs) - len(kept_blobs) - dropped_sparse > 0:
        warnings.append(
            f"Ignored {len(blobs) - len(kept_blobs) - dropped_sparse} oversized colored region(s) "
            "(likely a price badge/button, not a candle)."
        )
    if dropped_sparse > 0:
        warnings.append(
            f"Ignored {dropped_sparse} sparse/low-density region(s) "
            "(likely a gridline or UI artifact, not a real candle)."
        )
    blobs = kept_blobs

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
