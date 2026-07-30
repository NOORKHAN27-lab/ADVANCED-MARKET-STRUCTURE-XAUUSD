"""
app.py
------
AI Market Structure Analyzer — XAUUSD & BTCUSD
Developed by Noor Ahmed Khan.

Strategy pipeline (multi-timeframe smart-money style):
    1. HTF structure (BOS/CHOCH) → establishes directional bias
    2. LTF liquidity sweep aligned with that bias
    3. Supply/Demand zone formed right after the sweep
    4. Fibonacci retracement/extension for confirmation + targets

This tool surfaces trade *ideas* based on the rules above — it does not
place trades and is not financial advice.

Run locally:
    pip install streamlit yfinance requests pandas numpy matplotlib
    streamlit run app.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from data_feed import get_xauusd, get_btcusd
from structure import analyze_structure
from signal_engine import build_setups, win_rate_stats, win_rate_by_touch
from chart import plot_htf_structure, plot_ltf_setup, plot_winrate_gauge, plot_candles, style_axis
from image_extractor import extract_candles, COLOR_PRESETS

st.set_page_config(page_title="AI Market Structure Analyzer — by Noor Ahmed Khan",
                    page_icon="🧭", layout="wide")

# Fractal swing sensitivity is fixed internally (not user-configurable) --
# 2 bars left/right is a standard, reliable setting for this strategy.
FRACTAL_STRENGTH = 2

# Reference timezones shown alongside the NY session hours, so it's clear
# what time that session actually falls at around the world.
_TZ_REFERENCE = [
    ("Pakistan (PKT)", "Asia/Karachi"),
    ("UTC", "UTC"),
    ("London (UK)", "Europe/London"),
    ("Dubai (UAE)", "Asia/Dubai"),
]


def format_session_timezones(start_hour: int, end_hour: int) -> str:
    """Build an HTML snippet showing the NY session window converted into
    a few other timezones, so it's clear what local time that is."""
    today = date.today()
    ny_tz = ZoneInfo("America/New_York")
    start_dt = datetime(today.year, today.month, today.day, start_hour % 24, 0, tzinfo=ny_tz)
    end_dt = datetime(today.year, today.month, today.day, 0, 0, tzinfo=ny_tz) + timedelta(hours=end_hour)

    rows = []
    for label, tzname in _TZ_REFERENCE:
        tz = ZoneInfo(tzname)
        s_local = start_dt.astimezone(tz)
        e_local = end_dt.astimezone(tz)
        day_note = " (+1 day)" if e_local.date() > s_local.date() else ""
        rows.append(
            f'<div class="tz-row"><span class="tz-name">{label}</span>'
            f'<span class="tz-time">{s_local.strftime("%I:%M %p").lstrip("0")} '
            f'– {e_local.strftime("%I:%M %p").lstrip("0")}{day_note}</span></div>'
        )
    return "".join(rows)


# Timezones the user can pick from to enter their own local session hours;
# these get auto-converted to the equivalent New York (ET) hours internally.
TZ_OPTIONS = {
    "Pakistan (GMT+5)": "Asia/Karachi",
    "UTC (GMT+0)": "UTC",
    "London, UK": "Europe/London",
    "Dubai (GMT+4)": "Asia/Dubai",
    "New York (ET) — no conversion": "America/New_York",
}


def local_session_to_ny_hours(local_start: int, local_end: int, tz_name: str) -> tuple:
    """
    Converts a session window given in the user's own local time into the
    equivalent New York (ET) hour range used internally for filtering.
    Handles a session that wraps past local midnight (e.g. Pakistan
    5 PM - 2 AM) by treating `local_end` as the next day when it's
    numerically <= `local_start`.
    """
    today = date.today()
    local_tz = ZoneInfo(tz_name)
    ny_tz = ZoneInfo("America/New_York")

    start_dt = datetime(today.year, today.month, today.day, local_start % 24, 0, tzinfo=local_tz)
    end_hour_adjusted = local_end if local_end > local_start else local_end + 24
    end_dt = datetime(today.year, today.month, today.day, 0, 0, tzinfo=local_tz) + timedelta(hours=end_hour_adjusted)

    ny_start_dt = start_dt.astimezone(ny_tz)
    ny_end_dt = ny_start_dt + (end_dt - start_dt)

    ny_start_hour = ny_start_dt.hour
    ny_end_hour = ny_end_dt.hour if ny_end_dt.hour != 0 else 24
    return ny_start_hour, ny_end_hour

# ---------------------------------------------------------------------------
# Theme: "smart money" — deep indigo/near-black, violet accent (institutional
# order-flow feel), amber for demand zones, purple for supply zones.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Work+Sans:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap');

:root {
    --bg:       #0B0A14;
    --surface:  #14121F;
    --surface2: #1B1830;
    --border:   #2A2740;
    --text:     #EDEBF7;
    --muted:    #8A8FA3;
    --violet:   #9C8FF0;
    --violet-dim: #5D4FA0;
    --green:    #3FD08C;
    --red:      #E0576B;
    --amber:    #C7A15A;
}

.stApp {
    background:
        radial-gradient(ellipse 900px 480px at 90% -10%, rgba(156,143,240,0.10), transparent),
        radial-gradient(ellipse 700px 460px at 0% 100%, rgba(199,161,90,0.05), transparent),
        var(--bg);
    font-family: 'Work Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 4px 18px 4px; border-bottom: 1px solid var(--border); margin-bottom: 24px;
}
.navbar .brand { display: flex; align-items: center; gap: 10px; }
.navbar .logo {
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient(135deg, var(--violet), var(--violet-dim));
    display: flex; align-items: center; justify-content: center; font-size: 17px;
    box-shadow: 0 0 14px rgba(156,143,240,0.35);
}
.navbar .wordmark { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 18px; letter-spacing: -0.01em; }
.navbar .wordmark .accent {
    background: linear-gradient(90deg, var(--violet), #C9A9F5);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.pro-badge {
    font-family: 'Fira Code', monospace; font-size: 9.5px; font-weight: 500;
    padding: 2px 7px; border-radius: 4px; margin-left: 8px; letter-spacing: 0.06em;
    background: linear-gradient(90deg, var(--violet), var(--amber));
    color: #0B0A14;
}
.navbar .credit { font-family: 'Fira Code', monospace; font-size: 12px; color: var(--muted); text-align: right; }
.navbar .credit b { color: var(--violet); font-weight: 500; }

.hero {
    border: 1px solid var(--border); border-radius: 14px;
    background: linear-gradient(145deg, rgba(27,24,48,0.9), rgba(20,18,31,0.95));
    backdrop-filter: blur(10px);
    padding: 28px 34px 32px; margin-bottom: 24px; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
}
.hero::after {
    content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--violet), var(--amber), transparent);
    opacity: 0.8;
}
.hero::before {
    content: ""; position: absolute; top: -60%; right: -10%; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(156,143,240,0.15), transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 29px;
    margin: 0 0 8px 0; letter-spacing: -0.01em;
}
.hero h1 .accent {
    background: linear-gradient(90deg, var(--violet), #C9A9F5, var(--amber));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p.sub { font-size: 14px; color: var(--muted); max-width: 720px; margin: 0; line-height: 1.55; }

.section-label {
    font-family: 'Fira Code', monospace; font-size: 11.5px; letter-spacing: 0.08em;
    color: var(--violet); text-transform: uppercase; margin: 4px 0 10px 2px;
}

.stat-row { display: flex; gap: 12px; margin: 6px 0 18px 0; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 130px;
    background: linear-gradient(160deg, rgba(27,24,48,0.85), rgba(20,18,31,0.9));
    border: 1px solid var(--border); border-radius: 12px;
    padding: 15px 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
.stat-card .num { font-family: 'Fira Code', monospace; font-weight: 500; font-size: 23px; line-height: 1; }
.stat-card .lbl {
    font-family: 'Fira Code', monospace; font-size: 10.5px; letter-spacing: 0.05em;
    color: var(--muted); text-transform: uppercase; margin-top: 6px;
}
.stat-card.pos .num  { color: var(--green); text-shadow: 0 0 12px rgba(63,208,140,0.4); }
.stat-card.neg .num  { color: var(--red); text-shadow: 0 0 12px rgba(224,87,107,0.4); }
.stat-card.neutral .num { color: var(--text); }
.stat-card.gold .num { color: var(--amber); text-shadow: 0 0 12px rgba(199,161,90,0.4); }

.gauge-card {
    background: linear-gradient(160deg, rgba(27,24,48,0.85), rgba(20,18,31,0.9));
    border: 1px solid var(--border); border-radius: 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03);
    padding: 6px; display: flex; align-items: center; justify-content: center;
}

.bias-card {
    border-radius: 12px; border: 1px solid var(--border); padding: 18px 22px;
    display: flex; align-items: center; gap: 16px; margin-bottom: 22px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
.bias-card.bullish { background: rgba(63,208,140,0.08); border-color: rgba(63,208,140,0.35); }
.bias-card.bearish { background: rgba(224,87,107,0.08); border-color: rgba(224,87,107,0.35); }
.bias-card.neutral { background: var(--surface); }
.bias-card .dot { width: 14px; height: 14px; border-radius: 50%; }
.bias-card.bullish .dot { background: var(--green); box-shadow: 0 0 10px var(--green); }
.bias-card.bearish .dot { background: var(--red); box-shadow: 0 0 10px var(--red); }
.bias-card.neutral .dot { background: var(--muted); }
.bias-card .label { font-family: 'Fira Code', monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; }
.bias-card .value { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 20px; }
.bias-card.bullish .value { color: var(--green); }
.bias-card.bearish .value { color: var(--red); }

.setup-card {
    border: 1px solid var(--border); border-radius: 12px;
    background: linear-gradient(160deg, rgba(27,24,48,0.85), rgba(20,18,31,0.9));
    padding: 16px 20px; margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: transform 0.15s ease;
}
.setup-card:hover { transform: translateY(-1px); }
.setup-card .dir-buy {
    color: var(--green); font-family: 'Sora', sans-serif; font-weight: 700;
    text-shadow: 0 0 10px rgba(63,208,140,0.35);
}
.setup-card .dir-sell {
    color: var(--red); font-family: 'Sora', sans-serif; font-weight: 700;
    text-shadow: 0 0 10px rgba(224,87,107,0.35);
}
.rr-bar-track {
    width: 100%; height: 5px; border-radius: 3px; background: var(--surface2);
    margin: 8px 0 2px 0; overflow: hidden;
}
.rr-bar-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, var(--violet), var(--amber));
}
.setup-card .kv { font-family: 'Fira Code', monospace; font-size: 12.5px; color: var(--text); margin: 3px 0; }
.setup-card .kv .lbl { color: var(--muted); }
.setup-card .note { font-size: 12.5px; color: var(--muted); margin: 2px 0; padding-left: 14px; position: relative; }
.setup-card .note::before { content: "›"; position: absolute; left: 0; color: var(--violet); }
.confirmed-badge {
    display: inline-block; font-family: 'Fira Code', monospace; font-size: 10.5px;
    padding: 2px 8px; border-radius: 5px; margin-left: 8px;
}
.confirmed-badge.yes { background: rgba(63,208,140,0.15); color: var(--green); border: 1px solid rgba(63,208,140,0.35); }
.confirmed-badge.no { background: rgba(139,152,168,0.15); color: var(--muted); border: 1px solid rgba(139,152,168,0.35); }

.touch-badge {
    display: inline-block; font-family: 'Sora', sans-serif; font-weight: 700; font-size: 11px;
    padding: 2px 9px; border-radius: 5px; margin-right: 8px; vertical-align: middle;
}
.touch-badge.touch-a { background: rgba(156,143,240,0.15); color: var(--violet); border: 1px solid rgba(156,143,240,0.35); }
.touch-badge.touch-plus { background: rgba(199,161,90,0.18); color: var(--amber); border: 1px solid rgba(199,161,90,0.4); }

[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .section-label { color: var(--violet); }

.tz-panel {
    background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 12px; margin: 4px 0 12px 0;
}
.tz-row {
    display: flex; justify-content: space-between; align-items: center;
    font-family: 'Fira Code', monospace; font-size: 11px; padding: 3px 0;
}
.tz-name { color: var(--muted); }
.tz-time { color: var(--violet); font-weight: 500; }

.live-price-card {
    background: linear-gradient(160deg, rgba(27,24,48,0.85), rgba(20,18,31,0.9));
    border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; height: 100%;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03);
}
.live-label {
    font-family: 'Fira Code', monospace; font-size: 10.5px; letter-spacing: 0.08em;
    color: var(--muted); text-transform: uppercase; margin-bottom: 6px;
}
.live-label::before {
    content: "●"; color: var(--green); margin-right: 6px; font-size: 9px;
    animation: blink 1.6s ease-in-out infinite;
}
.live-price {
    font-family: 'Fira Code', monospace; font-weight: 500; font-size: 28px; color: var(--text);
}
.live-change { font-family: 'Fira Code', monospace; font-size: 13px; margin-top: 4px; }
.live-change.pos { color: var(--green); }
.live-change.neg { color: var(--red); }

.touch-compare { display: flex; gap: 12px; margin: 4px 0 20px 0; }
.touch-card {
    flex: 1; text-align: center;
    background: linear-gradient(160deg, rgba(27,24,48,0.85), rgba(20,18,31,0.9));
    border: 1px solid var(--border); border-radius: 12px; padding: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.touch-card.plus { border-color: rgba(199,161,90,0.4); }
.touch-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 16px; color: var(--text); }
.touch-title .touch-sub { font-family: 'Fira Code', monospace; font-size: 10.5px; color: var(--muted); font-weight: 400; }
.touch-wr { font-family: 'Fira Code', monospace; font-size: 30px; font-weight: 500; margin: 6px 0; color: var(--violet); }
.touch-card.plus .touch-wr { color: var(--amber); }
.touch-detail { font-family: 'Fira Code', monospace; font-size: 11px; color: var(--muted); }

.stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
    background: var(--surface2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 7px !important;
    font-family: 'Fira Code', monospace !important;
}
.stButton > button {
    font-family: 'Work Sans', sans-serif; font-weight: 600; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface2); color: var(--text);
}
.stButton > button:hover { border-color: var(--violet); color: var(--violet); }
.stButton > button[kind="primary"] { background: var(--violet); color: #0B0A14; border: none; }
.stButton > button[kind="primary"]:hover { background: #B4A9F5; color: #0B0A14; }

[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 10px !important; background: var(--surface) !important; }
.stAlert { border-radius: 8px !important; font-family: 'Fira Code', monospace; }

.footer-note {
    font-family: 'Fira Code', monospace; font-size: 11.5px; color: var(--muted);
    text-align: center; margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--border);
}
.footer-note .credit { color: var(--violet); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navbar + hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="brand">
        <div class="logo">🧭</div>
        <div class="wordmark">Market Structure <span class="accent">Analyzer</span><span class="pro-badge">PRO</span></div>
    </div>
    <div class="credit">Developed by <b>Noor Ahmed Khan</b></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>AI Market <span class="accent">Structure</span> Analyzer</h1>
    <p class="sub">Multi-timeframe smart-money analysis for XAUUSD and BTCUSD: HTF structure
    (BOS/CHOCH) sets the bias, then the LTF is scanned for a liquidity sweep, a supply/demand
    zone, and Fibonacci confirmation — the exact sequence of your strategy. This surfaces
    trade ideas for you to evaluate; it does not place trades and is not financial advice.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
    input_mode = st.radio("Mode", ["Live Market Data", "Upload Chart Image"], index=0, label_visibility="collapsed")
    use_image_mode = input_mode == "Upload Chart Image"

    st.markdown('<div class="section-label">Market</div>', unsafe_allow_html=True)
    symbol_label = st.selectbox("Symbol", ["XAUUSD (Gold)", "BTCUSD (Bitcoin)"])

    uploaded_image = None
    bull_color = bear_color = None
    top_price = bottom_price = None
    chart_tf_label = "1h"

    if use_image_mode:
        st.markdown('<div class="section-label">Chart Image</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload a chart screenshot", type=["png", "jpg", "jpeg"])
        chart_tf_label = st.selectbox("This chart's timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)

        preset_label = st.selectbox("Candle colors", list(COLOR_PRESETS.keys()), index=0)
        if COLOR_PRESETS[preset_label] is None:
            c1, c2 = st.columns(2)
            with c1:
                bull_hex = st.color_picker("Bull (up) color", "#089981")
            with c2:
                bear_hex = st.color_picker("Bear (down) color", "#F23645")
            bull_color = tuple(int(bull_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            bear_color = tuple(int(bear_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        else:
            bull_color, bear_color = COLOR_PRESETS[preset_label]

        st.caption("Price axis is read automatically from your chart — no need to type it in.")
        with st.expander("Manual price override (only if auto-detect fails)"):
            use_manual_calibration = st.checkbox("Enter prices manually instead", value=False)
            if use_manual_calibration:
                p1, p2 = st.columns(2)
                with p1:
                    top_price = st.number_input("Price at TOP of image", min_value=0.0, value=2050.0, step=0.5)
                with p2:
                    bottom_price = st.number_input("Price at BOTTOM of image", min_value=0.0, value=1950.0, step=0.5)

        htf_label = ltf_label = chart_tf_label
    else:
        st.markdown('<div class="section-label">Timeframes</div>', unsafe_allow_html=True)
        htf_label = st.selectbox("HTF (bias / structure)", ["1d", "4h", "1h"], index=0)
        ltf_label = st.selectbox("LTF (sweep / entry)", ["30m", "15m", "5m", "1m"], index=0)

    st.markdown('<div class="section-label">Execution Rules</div>', unsafe_allow_html=True)
    session_only = st.checkbox("Only NY session sweeps", value=True, disabled=use_image_mode)
    if use_image_mode:
        st.caption("Session filtering needs real timestamps, which a screenshot doesn't have — disabled in this mode.")
        session_only = False

    tz_label = st.selectbox("Your timezone", list(TZ_OPTIONS.keys()), index=0, disabled=not session_only)
    tz_name = TZ_OPTIONS[tz_label]

    local_col1, local_col2 = st.columns(2)
    with local_col1:
        local_start = st.number_input(f"Session start ({tz_label.split(' ')[0]} time)",
                                       min_value=0, max_value=23, value=17, disabled=not session_only)
    with local_col2:
        local_end = st.number_input(f"Session end ({tz_label.split(' ')[0]} time)",
                                     min_value=0, max_value=23, value=2, disabled=not session_only)

    if session_only:
        session_start, session_end = local_session_to_ny_hours(int(local_start), int(local_end), tz_name)
        st.markdown(
            f'<div class="tz-panel"><div class="tz-row">'
            f'<span class="tz-name">= NY session (ET)</span>'
            f'<span class="tz-time">{session_start:02d}:00 – {session_end:02d}:00</span>'
            f'</div>{format_session_timezones(session_start, session_end)}</div>',
            unsafe_allow_html=True,
        )
    else:
        session_start, session_end = 8, 17

    fresh_only = st.checkbox("Fresh (untested) zones only", value=True)

    rr_col1, rr_col2 = st.columns(2)
    with rr_col1:
        min_rr = st.number_input("Min R:R", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
    with rr_col2:
        max_rr = st.number_input("Max R:R", min_value=0.5, max_value=20.0, value=5.0, step=0.5)

    run = st.button(
        "🔍  Extract & Analyze Chart" if use_image_mode else "▶  Run Analysis",
        type="primary", use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Live price snapshot — updates the moment the symbol dropdown changes,
# independent of "Run Analysis". Cached briefly so switching symbols back
# and forth doesn't re-hit the API every single time.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def _get_live_snapshot(symbol_label: str):
    if symbol_label.startswith("XAUUSD"):
        return get_xauusd(interval="5m", lookback_days=5)
    return get_btcusd(interval="5m", limit=200)


live_data = _get_live_snapshot(symbol_label)

if live_data is not None and len(live_data) > 1:
    last_price = float(live_data["Close"].iloc[-1])
    prev_price = float(live_data["Close"].iloc[-2])
    change = last_price - prev_price
    change_pct = (change / prev_price * 100) if prev_price else 0.0
    change_class = "pos" if change >= 0 else "neg"
    arrow = "▲" if change >= 0 else "▼"
    price_fmt = f"{last_price:,.2f}"

    live_col1, live_col2 = st.columns([1, 3])
    with live_col1:
        st.markdown(f"""
        <div class="live-price-card">
            <div class="live-label">{symbol_label} — LIVE (5m)</div>
            <div class="live-price">{price_fmt}</div>
            <div class="live-change {change_class}">{arrow} {change:+.2f} ({change_pct:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            _get_live_snapshot.clear()
            st.rerun()

    with live_col2:
        fig_live, ax_live = plt.subplots(figsize=(10, 3.2))
        fig_live.patch.set_facecolor("#14121F")
        plot_candles(ax_live, live_data.tail(80))
        style_axis(ax_live, f"{symbol_label} — Recent Price Action")
        st.pyplot(fig_live)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    if use_image_mode:
        if uploaded_image is None:
            st.error("Please upload a chart screenshot first.")
            st.stop()

        with st.spinner("Extracting candles from the image..."):
            from PIL import Image as PILImage
            img = PILImage.open(uploaded_image)
            extraction = extract_candles(
                img, bull_color, bear_color, top_price, bottom_price,
                color_tolerance=30, interval=chart_tf_label,
            )

        for w in extraction.warnings:
            if w.startswith("Auto-detected"):
                st.info(w)
            else:
                st.warning(w)

        if extraction.candle_count == 0:
            st.error("No candles could be detected in this image. Try adjusting the "
                      "candle colors, or upload a clearer/cropped screenshot.")
            st.stop()

        st.success(f"Extracted {extraction.candle_count} candles from the image.")
        htf_data = ltf_data = extraction.data

        with st.spinner("Analyzing market structure..."):
            htf_swings, htf_events, bias = analyze_structure(
                htf_data, left=FRACTAL_STRENGTH, right=FRACTAL_STRENGTH)
            setups = build_setups(
                ltf_data, bias, ltf_left=FRACTAL_STRENGTH, ltf_right=FRACTAL_STRENGTH,
                session_only=False, fresh_zones_only=fresh_only, min_rr=min_rr, max_rr=max_rr,
            )
            stats = win_rate_stats(setups)

        st.markdown('<div class="section-label">Extracted Chart (verify this matches your screenshot)</div>',
                     unsafe_allow_html=True)
        fig_extract, ax_extract = plt.subplots(figsize=(13, 5))
        fig_extract.patch.set_facecolor("#14121F")
        plot_candles(ax_extract, htf_data)
        style_axis(ax_extract, f"Reconstructed from image — {symbol_label} ({chart_tf_label})")
        st.pyplot(fig_extract)

    else:
        # yfinance limits how far back intraday data goes: 1m ~7 days, 5m/15m/30m ~60 days
        ltf_lookback_map = {"1m": 7, "5m": 60, "15m": 60, "30m": 60}
        ltf_lookback = ltf_lookback_map.get(ltf_label, 30)

        with st.spinner("Fetching data and analyzing market structure..."):
            if symbol_label.startswith("XAUUSD"):
                htf_data = get_xauusd(interval=htf_label, lookback_days=180)
                ltf_data = get_xauusd(interval=ltf_label, lookback_days=ltf_lookback)
            else:
                htf_data = get_btcusd(interval=htf_label, limit=500)
                ltf_data = get_btcusd(interval=ltf_label, limit=1000)

            htf_swings, htf_events, bias = analyze_structure(
                htf_data, left=FRACTAL_STRENGTH, right=FRACTAL_STRENGTH)
            setups = build_setups(
                ltf_data, bias, ltf_left=FRACTAL_STRENGTH, ltf_right=FRACTAL_STRENGTH,
                session_only=session_only, session_start_hour=int(session_start), session_end_hour=int(session_end),
                fresh_zones_only=fresh_only, min_rr=min_rr, max_rr=max_rr,
            )
            stats = win_rate_stats(setups)

    st.markdown(f"""
    <div class="bias-card {bias}">
        <div class="dot"></div>
        <div>
            <div class="label">Current HTF Bias ({htf_label})</div>
            <div class="value">{bias.upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gauge_col, stats_col = st.columns([1, 2.4])
    with gauge_col:
        st.markdown('<div class="gauge-card">', unsafe_allow_html=True)
        gauge_fig = plot_winrate_gauge(stats["win_rate"], stats["wins"], stats["losses"])
        st.pyplot(gauge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with stats_col:
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card neutral"><div class="num">{stats['total_setups']}</div><div class="lbl">Setups Found</div></div>
            <div class="stat-card neutral"><div class="num">{stats['closed']}</div><div class="lbl">Closed (Win/Loss)</div></div>
        </div>
        <div class="stat-row">
            <div class="stat-card pos"><div class="num">{stats['wins']}</div><div class="lbl">Wins</div></div>
            <div class="stat-card neg"><div class="num">{stats['losses']}</div><div class="lbl">Losses</div></div>
            <div class="stat-card gold"><div class="num">{stats['open']}</div><div class="lbl">Open</div></div>
        </div>
        """, unsafe_allow_html=True)

    if stats['closed'] < 10:
        st.warning(f"Only {stats['closed']} closed setup(s) in this window — too small a sample to judge win rate. "
                   "Widen the LTF history (more bars / longer period) for a meaningful sample size.")

    touch_stats = win_rate_by_touch(setups)
    a_stats, a_plus_stats = touch_stats["A"], touch_stats["A+"]
    if a_stats["closed"] > 0 or a_plus_stats["closed"] > 0:
        st.markdown(f"""
        <div class="touch-compare">
            <div class="touch-card">
                <div class="touch-title">A <span class="touch-sub">(first touch)</span></div>
                <div class="touch-wr">{a_stats['win_rate']:.1f}%</div>
                <div class="touch-detail">{a_stats['wins']}W · {a_stats['losses']}L · {a_stats['total_setups']} setup(s)</div>
            </div>
            <div class="touch-card plus">
                <div class="touch-title">A+ <span class="touch-sub">(confirmed 2nd touch)</span></div>
                <div class="touch-wr">{a_plus_stats['win_rate']:.1f}%</div>
                <div class="touch-detail">{a_plus_stats['wins']}W · {a_plus_stats['losses']}L · {a_plus_stats['total_setups']} setup(s)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    tab_htf, tab_ltf, tab_setups = st.tabs(["HTF Structure", "LTF Setups", "Setup Details"])

    with tab_htf:
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor("#14121F")
        plot_htf_structure((fig, ax), htf_data.tail(200), [e for e in htf_events if e.index >= len(htf_data) - 200])
        st.pyplot(fig)
        st.caption(f"{len(htf_swings)} swing points detected · {len(htf_events)} structure breaks (BOS/CHOCH) total.")

    with tab_ltf:
        if setups:
            latest = setups[-1]
            fig2, ax2 = plt.subplots(figsize=(13, 6))
            fig2.patch.set_facecolor("#14121F")
            window = ltf_data.iloc[max(0, latest.zone.start_index - 20): latest.sweep.index + 40]
            plot_ltf_setup((fig2, ax2), window, latest.sweep, latest.zone, latest.fib_levels)
            st.pyplot(fig2)
            st.caption(f"Showing the most recent aligned setup. {len(setups)} total setup(s) found in this window.")
        else:
            st.info(f"No liquidity sweep + zone setup aligned with the {bias} bias was found in this LTF window. "
                    "Try a longer LTF history.")

    with tab_setups:
        if not setups:
            st.write("No setups to show yet — run analysis with more data.")
        for i, s in enumerate(reversed(setups), start=1):
            dir_class = "dir-buy" if s.direction == "BUY" else "dir-sell"
            outcome_class = {"WIN": "yes", "LOSS": "no", "OPEN": "no"}[s.outcome]
            outcome_text = {
                "WIN": "WIN — target hit",
                "LOSS": "LOSS — stop hit",
                "OPEN": "AWAITING RETURN / OPEN",
            }[s.outcome]
            touch_class = "touch-plus" if s.touch_label == "A+" else "touch-a"
            rr_pct = max(0.0, min(100.0, (s.risk_reward / max_rr) * 100))
            st.markdown(f"""
            <div class="setup-card">
                <span class="{dir_class}">{s.direction} SETUP</span>
                <span class="touch-badge {touch_class}">{s.touch_label}</span>
                <span class="confirmed-badge {outcome_class}">{outcome_text}</span>
                <div class="kv"><span class="lbl">Entry zone:</span> {s.entry_zone[0]:.2f} – {s.entry_zone[1]:.2f}</div>
                <div class="kv"><span class="lbl">Stop loss:</span> {s.stop_loss:.2f}</div>
                <div class="kv"><span class="lbl">Target used:</span> {s.chosen_target:.2f} (R:R = 1:{s.risk_reward:.2f})</div>
                <div class="rr-bar-track"><div class="rr-bar-fill" style="width:{rr_pct:.0f}%"></div></div>
                <div class="kv"><span class="lbl">Sweep date:</span> {pd.Timestamp(s.sweep.date):%Y-%m-%d %H:%M}</div>
                {f'<div class="kv"><span class="lbl">Bars held:</span> {s.bars_held}</div>' if s.bars_held is not None else ''}
                {''.join(f'<div class="note">{n}</div>' for n in s.confluence_notes)}
            </div>
            """, unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Educational tool based on historical/live market data — '
    'not financial advice, no trades are placed automatically.'
    '<br>AI Market Structure Analyzer — <span class="credit">Developed by Noor Ahmed Khan</span></div>',
    unsafe_allow_html=True,
)
