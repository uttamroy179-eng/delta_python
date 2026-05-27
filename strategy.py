# engine/strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# FIX #12 (from risk_manager): Import shared constants instead of redefining them
from engine.risk_manager import (
    ATR_SL_MULTIPLIER,
    TP1_PCT,
    TP2_PCT,
    TP3_PCT,
)

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------

EMA_SHORT           = 9
EMA_LONG            = 21
ATR_PERIOD          = 14
RSI_PERIOD          = 14
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
BB_PERIOD           = 20
BB_STD              = 2
STOCH_K_PERIOD      = 14
STOCH_D_PERIOD      = 3
SUPERTREND_PERIOD   = 10
SUPERTREND_MULT     = 3.0
VOLUME_SPIKE_MULT   = 2.0
BREAKOUT_PERIOD     = 20
ATR_TP_MULTIPLIER   = 4.5

# TP levels split across ATR_TP_MULTIPLIER
TP1_MULT = ATR_TP_MULTIPLIER / 3 * 1   # 1.5x ATR
TP2_MULT = ATR_TP_MULTIPLIER / 3 * 2   # 3.0x ATR
TP3_MULT = ATR_TP_MULTIPLIER           # 4.5x ATR

# Pyramid add-on trigger: price moves 0.5x ATR in favor
PYRAMID_ATR_TRIGGER = 0.5

# Multi-timeframe weights
MTF_WEIGHTS = {"5m": 0.25, "15m": 0.35}

# FIX (run_all_strategies): Minimum combined confidence to generate a signal
SIGNAL_THRESHOLD = 40.0

# EMA crossover: minimum gap between EMAs as % of price to avoid noise signals
EMA_MIN_GAP_PCT = 0.0005   # 0.05% of price

# RSI confirmation thresholds
RSI_BUY_MIN  = 45.0   # RSI must be above this for a BUY confirmation
RSI_BUY_MAX  = 65.0   # RSI must be below this for a BUY confirmation (not overbought)
RSI_SELL_MAX = 55.0   # RSI must be below this for a SELL confirmation (bearish momentum)

# SuperTrend: minimum number of bars the trend must have been active
# before a continuation signal is accepted (avoids whipsaw on fresh flips)
SUPERTREND_MIN_TREND_BARS = 2


# -----------------------------------------------
# DATAFRAME CONVERSION
# -----------------------------------------------

def convert_to_dataframe(data) -> pd.DataFrame:
    """
    Convert raw candle list or existing DataFrame to a
    standardized DataFrame with Title Case OHLCV columns.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)

    df.columns = [col.capitalize() for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"[Strategy] Missing required column: {col}")

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], unit="s")
        df.set_index("Time", inplace=True)

    df = df[required].astype(float)
    df.sort_index(inplace=True)
    return df


# -----------------------------------------------
# CORE INDICATORS
# -----------------------------------------------

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h_l  = df["High"] - df["Low"]
    h_pc = (df["High"] - df["Close"].shift(1)).abs()
    l_pc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# -----------------------------------------------
# STRATEGY 1: EMA CROSSOVER  (FIXED)
# -----------------------------------------------

def strategy_ema_crossover(df: pd.DataFrame) -> Dict[str, Any]:
    """
    EMA crossover strategy.

    FIX 1 — Trend-state approach:
        Signal is generated when EMAs cross AND the current bar confirms
        the new alignment. This avoids missing the exact crossover tick
        in live data while still requiring a fresh cross (not just alignment).

    FIX 2 — Minimum gap threshold:
        A cross is only accepted if the gap between EMAs exceeds
        EMA_MIN_GAP_PCT of the current price. This filters noise crosses
        where the two EMAs are essentially flat against each other.

    FIX 3 — Confidence scaling:
        Confidence is now proportional to the EMA gap relative to price,
        scaled so that a 0.1% gap = ~50 confidence and a 0.2% gap = ~100.
        Previously gap * 20 produced near-zero values.
    """
    ema_short = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    ema_long  = df["Close"].ewm(span=EMA_LONG,  adjust=False).mean()

    curr_short = ema_short.iloc[-1]
    curr_long  = ema_long.iloc[-1]
    prev_short = ema_short.iloc[-2]
    prev_long  = ema_long.iloc[-2]
    price      = df["Close"].iloc[-1]

    # Gap as a fraction of price
    gap_pct = abs(curr_short - curr_long) / (price + 1e-9)

    # FIX 2: Reject crosses where EMAs are too close together (noise)
    if gap_pct < EMA_MIN_GAP_PCT:
        return {
            "signal":     "HOLD",
            "confidence": 0.0,
            "ema_short":  round(float(curr_short), 4),
            "ema_long":   round(float(curr_long),  4),
            "gap_pct":    round(float(gap_pct * 100), 4),
            "reason":     "EMA gap below minimum threshold",
        }

    # FIX 1: Detect fresh crossover (state change on this bar)
    bullish_cross = (curr_short > curr_long) and (prev_short <= prev_long)
    bearish_cross = (curr_short < curr_long) and (prev_short >= prev_long)

    # FIX 3: Scale confidence — 0.05% gap -> ~50, 0.2% gap -> ~100
    confidence = round(float(min(gap_pct / 0.002 * 100, 100)), 2)

    if bullish_cross:
        signal = "BUY"
    elif bearish_cross:
        signal = "SELL"
    else:
        signal     = "HOLD"
        confidence = 0.0

    return {
        "signal":     signal,
        "confidence": confidence,
        "ema_short":  round(float(curr_short), 4),
        "ema_long":   round(float(curr_long),  4),
        "gap_pct":    round(float(gap_pct * 100), 4),
    }


# -----------------------------------------------
# STRATEGY 2: SUPERTREND  (FIXED)
# -----------------------------------------------

def strategy_supertrend(df: pd.DataFrame) -> Dict[str, Any]:
    """
    SuperTrend indicator strategy.

    FIX 1 — pandas >= 2.0 compatibility:
        Replaced .iloc[i] = inside loop with numpy arrays to avoid
        ChainedAssignmentError and achieve ~10x speedup on large DataFrames.

    FIX 2 — NaN at index 0:
        Index 0 is now explicitly seeded before the loop starts.

    FIX 3 — Dynamic confidence:
        Confidence is now based on how far price is from the SuperTrend line
        relative to ATR, giving a meaningful 0-100 score.

    FIX 4 — Trend continuation accepted:
        Signal fires on the crossover bar AND on continuation bars where
        the trend has been active for at least SUPERTREND_MIN_TREND_BARS bars.
        This ensures the bot can enter even if it missed the exact flip bar.
    """
    n      = len(df)
    atr    = calculate_atr(df, SUPERTREND_PERIOD).values
    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    hl2    = (high + low) / 2.0

    # FIX 1: Use numpy arrays instead of pandas Series for the loop
    upper      = np.zeros(n)
    lower      = np.zeros(n)
    supertrend = np.zeros(n)
    direction  = np.zeros(n, dtype=int)   # 1 = bullish, -1 = bearish

    # Raw bands
    raw_upper = hl2 + SUPERTREND_MULT * atr
    raw_lower = hl2 - SUPERTREND_MULT * atr

    # FIX 2: Seed index 0 explicitly to avoid NaN on first loop iteration
    upper[0]      = raw_upper[0]
    lower[0]      = raw_lower[0]
    supertrend[0] = raw_upper[0]
    direction[0]  = -1   # Start bearish until proven otherwise

    for i in range(1, n):
        # Upper band: only tighten (move down) if previous close was below it
        upper[i] = (
            min(raw_upper[i], upper[i - 1])
            if close[i - 1] <= upper[i - 1]
            else raw_upper[i]
        )

        # Lower band: only tighten (move up) if previous close was above it
        lower[i] = (
            max(raw_lower[i], lower[i - 1])
            if close[i - 1] >= lower[i - 1]
            else raw_lower[i]
        )

        # Determine direction based on previous supertrend value
        if supertrend[i - 1] == upper[i - 1]:
            # Was bearish
            if close[i] <= upper[i]:
                supertrend[i] = upper[i]
                direction[i]  = -1
            else:
                supertrend[i] = lower[i]
                direction[i]  = 1
        else:
            # Was bullish
            if close[i] >= lower[i]:
                supertrend[i] = lower[i]
                direction[i]  = 1
            else:
                supertrend[i] = upper[i]
                direction[i]  = -1

    curr_dir  = int(direction[-1])
    prev_dir  = int(direction[-2])
    is_flip   = curr_dir != prev_dir

    # FIX 4: Count how many consecutive bars the current trend has been active
    trend_bars = 0
    for i in range(n - 1, -1, -1):
        if direction[i] == curr_dir:
            trend_bars += 1
        else:
            break

    # Accept signal on flip bar OR if trend has been running long enough
    trend_is_valid = is_flip or (trend_bars >= SUPERTREND_MIN_TREND_BARS)

    # FIX 3: Dynamic confidence based on distance of price from SuperTrend line
    # relative to ATR. Capped at 100.
    current_atr = float(atr[-1]) if float(atr[-1]) > 0 else 1.0
    price_dist  = abs(float(close[-1]) - float(supertrend[-1]))
    confidence  = round(float(min((price_dist / current_atr) * 50, 100)), 2)

    if curr_dir == 1 and trend_is_valid:
        signal = "BUY"
    elif curr_dir == -1 and trend_is_valid:
        signal = "SELL"
    else:
        signal     = "HOLD"
        confidence = 0.0

    return {
        "signal":      signal,
        "confidence":  confidence,
        "supertrend":  round(float(supertrend[-1]), 4),
        "direction":   curr_dir,
        "trend_bars":  trend_bars,
        "is_flip":     is_flip,
    }


# -----------------------------------------------
# STRATEGY 3: RSI CONFIRMATION  (FIXED)
# -----------------------------------------------

def strategy_rsi_confirmation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    RSI-based confirmation strategy.

    Replaces strategy_rsi_divergence. The divergence check over only
    2 bars (iloc[-1] vs iloc[-3]) was not meaningful. This version uses
    RSI as a momentum filter and zone detector:

    FIX 1 — Divergence lookback:
        Bullish/bearish divergence is now detected over a configurable
        lookback window (default 10 bars) by comparing the most recent
        swing low/high in price vs RSI.

    FIX 2 — Confidence formula:
        BUY confidence is proportional to how far RSI is below 50
        (momentum building from oversold). SELL confidence is proportional
        to how far RSI is above 50. Both are scaled to 0-100.

    FIX 3 — RSI gate thresholds:
        Exported RSI_BUY_MIN, RSI_BUY_MAX, RSI_SELL_MAX are used by
        run_all_strategies to gate the combined signal. These replace
        the broken `rsi < 50 or rsi > 65` SELL condition.

    Signal logic:
        BUY  : RSI < 50 (building momentum) OR bullish divergence detected
        SELL : RSI > 50 (fading momentum)   OR bearish divergence detected
        HOLD : RSI near 50 with no divergence
    """
    rsi      = calculate_rsi(df["Close"], RSI_PERIOD)
    rsi_vals = rsi.values
    close    = df["Close"].values
    last_rsi = float(rsi_vals[-1])

    # FIX 1: Divergence over a meaningful lookback window
    lookback = min(10, len(df) - 2)

    # Bullish divergence: price makes lower low, RSI makes higher low
    price_recent_low = float(close[-1])
    price_past_low   = float(np.min(close[-(lookback + 1):-1]))
    rsi_recent_low   = last_rsi
    rsi_past_low     = float(np.min(rsi_vals[-(lookback + 1):-1]))

    bullish_divergence = (
        price_recent_low < price_past_low   # price lower low
        and rsi_recent_low > rsi_past_low   # RSI higher low
        and last_rsi < 50                   # still in bearish RSI territory
    )

    # Bearish divergence: price makes higher high, RSI makes lower high
    price_recent_high = float(close[-1])
    price_past_high   = float(np.max(close[-(lookback + 1):-1]))
    rsi_recent_high   = last_rsi
    rsi_past_high     = float(np.max(rsi_vals[-(lookback + 1):-1]))

    bearish_divergence = (
        price_recent_high > price_past_high   # price higher high
        and rsi_recent_high < rsi_past_high   # RSI lower high
        and last_rsi > 50                     # still in bullish RSI territory
    )

    # FIX 2: Confidence proportional to RSI distance from 50
    if last_rsi < 40 or bullish_divergence:
        signal     = "BUY"
        # Stronger signal the further RSI is below 50
        confidence = round(float(min((50 - last_rsi) * 2, 100)), 2)
        confidence = max(confidence, 30.0)   # Floor at 30 for divergence cases

    elif last_rsi > 60 or bearish_divergence:
        signal     = "SELL"
        # Stronger signal the further RSI is above 50
        confidence = round(float(min((last_rsi - 50) * 2, 100)), 2)
        confidence = max(confidence, 30.0)

    else:
        signal     = "HOLD"
        confidence = 0.0

    return {
        "signal":              signal,
        "confidence":          confidence,
        "rsi":                 round(last_rsi, 2),
        "bullish_divergence":  bullish_divergence,
        "bearish_divergence":  bearish_divergence,
    }


# -----------------------------------------------
# OTHER STRATEGIES (unchanged)
# -----------------------------------------------

def strategy_macd(df: pd.DataFrame) -> Dict[str, Any]:
    ema_fast    = df["Close"].ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow    = df["Close"].ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL,   adjust=False).mean()
    histogram   = macd_line - signal_line
    prev = histogram.iloc[-2]
    curr = histogram.iloc[-1]
    if curr > 0 and prev <= 0:
        signal     = "BUY"
        confidence = min(abs(curr) / (df["Close"].iloc[-1] * 0.001) * 100, 100)
    elif curr < 0 and prev >= 0:
        signal     = "SELL"
        confidence = min(abs(curr) / (df["Close"].iloc[-1] * 0.001) * 100, 100)
    else:
        signal     = "HOLD"
        confidence = 0.0
    return {
        "signal":      signal,
        "confidence":  round(float(confidence), 2),
        "macd":        round(float(macd_line.iloc[-1]),   4),
        "macd_signal": round(float(signal_line.iloc[-1]), 4),
        "histogram":   round(float(histogram.iloc[-1]),   4),
    }


def strategy_bollinger_bands(df: pd.DataFrame) -> Dict[str, Any]:
    sma   = df["Close"].rolling(BB_PERIOD).mean()
    std   = df["Close"].rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    close = df["Close"].iloc[-1]
    width = upper.iloc[-1] - lower.iloc[-1]
    if close <= lower.iloc[-1]:
        signal     = "BUY"
        confidence = min((lower.iloc[-1] - close) / (width + 1e-9) * 200, 100)
    elif close >= upper.iloc[-1]:
        signal     = "SELL"
        confidence = min((close - upper.iloc[-1]) / (width + 1e-9) * 200, 100)
    else:
        signal     = "HOLD"
        confidence = 0.0
    return {
        "signal":     signal,
        "confidence": round(float(confidence), 2),
        "bb_upper":   round(float(upper.iloc[-1]), 4),
        "bb_lower":   round(float(lower.iloc[-1]), 4),
        "bb_mid":     round(float(sma.iloc[-1]),   4),
    }


def strategy_stochastic(df: pd.DataFrame) -> Dict[str, Any]:
    low_min  = df["Low"].rolling(STOCH_K_PERIOD).min()
    high_max = df["High"].rolling(STOCH_K_PERIOD).max()
    k        = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    d        = k.rolling(STOCH_D_PERIOD).mean()
    k_curr, d_curr = k.iloc[-1], d.iloc[-1]
    k_prev, d_prev = k.iloc[-2], d.iloc[-2]
    if k_curr < 20 and d_curr < 20 and k_curr > d_curr and k_prev <= d_prev:
        signal     = "BUY"
        confidence = round(float((20 - k_curr) * 2.5), 2)
    elif k_curr > 80 and d_curr > 80 and k_curr < d_curr and k_prev >= d_prev:
        signal     = "SELL"
        confidence = round(float((k_curr - 80) * 2.5), 2)
    else:
        signal     = "HOLD"
        confidence = 0.0
    return {
        "signal":     signal,
        "confidence": min(confidence, 100) if signal != "HOLD" else 0.0,
        "stoch_k":    round(float(k_curr), 2),
        "stoch_d":    round(float(d_curr), 2),
    }


def strategy_volume_spike(df: pd.DataFrame) -> Dict[str, Any]:
    avg_vol    = df["Volume"].rolling(20).mean()
    curr_vol   = df["Volume"].iloc[-1]
    avg        = avg_vol.iloc[-1]
    spike      = curr_vol > (VOLUME_SPIKE_MULT * avg)
    price_up   = df["Close"].iloc[-1] > df["Open"].iloc[-1]
    price_down = df["Close"].iloc[-1] < df["Open"].iloc[-1]
    vol_ratio  = curr_vol / (avg + 1e-9)
    if spike and price_up:
        signal     = "BUY"
        confidence = round(float(min((vol_ratio - 1) * 30, 100)), 2)
    elif spike and price_down:
        signal     = "SELL"
        confidence = round(float(min((vol_ratio - 1) * 30, 100)), 2)
    else:
        signal     = "HOLD"
        confidence = 0.0
    return {
        "signal":     signal,
        "confidence": confidence,
        "vol_ratio":  round(float(vol_ratio), 2),
        "volume":     round(float(curr_vol),  2),
        "avg_volume": round(float(avg),       2),
    }


def strategy_breakout(df: pd.DataFrame) -> Dict[str, Any]:
    highest    = df["High"].rolling(BREAKOUT_PERIOD).max()
    lowest     = df["Low"].rolling(BREAKOUT_PERIOD).min()
    close      = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    width      = highest.iloc[-1] - lowest.iloc[-1]
    if prev_close < highest.iloc[-2] and close >= highest.iloc[-1]:
        signal     = "BUY"
        confidence = round(float(min((close - highest.iloc[-2]) / (width + 1e-9) * 200, 100)), 2)
    elif prev_close > lowest.iloc[-2] and close <= lowest.iloc[-1]:
        signal     = "SELL"
        confidence = round(float(min((lowest.iloc[-2] - close) / (width + 1e-9) * 200, 100)), 2)
    else:
        signal     = "HOLD"
        confidence = 0.0
    return {
        "signal":        signal,
        "confidence":    confidence,
        "breakout_high": round(float(highest.iloc[-1]), 4),
        "breakout_low":  round(float(lowest.iloc[-1]),  4),
    }


# -----------------------------------------------
# MULTI-TIMEFRAME AGGREGATION
# -----------------------------------------------

def aggregate_mtf_signals(mtf_results: Dict[str, Dict]) -> Dict[str, Any]:
    score        = 0.0
    total_weight = 0.0
    for tf, result in mtf_results.items():
        weight = MTF_WEIGHTS.get(tf, 0.33)
        sig    = result.get("signal", "HOLD")
        conf   = result.get("confidence", 0.0)
        if sig == "BUY":
            score += weight * conf
        elif sig == "SELL":
            score -= weight * conf
        total_weight += weight
    normalized = score / (total_weight + 1e-9)
    if normalized >= 40:
        signal     = "BUY"
        confidence = round(float(min(normalized, 100)), 2)
    elif normalized <= -40:
        signal     = "SELL"
        confidence = round(float(min(abs(normalized), 100)), 2)
    else:
        signal     = "HOLD"
        confidence = 0.0
    return {
        "signal":     signal,
        "confidence": confidence,
        "mtf_score":  round(float(normalized), 2),
        "timeframes": mtf_results,
    }


# -----------------------------------------------
# COMBINED SIGNAL GENERATOR  (FIXED)
# -----------------------------------------------

def run_all_strategies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all active strategies and return a combined weighted signal.

    FIX (scoring loop):
        Previously the RSI gate was checked INSIDE the per-strategy loop,
        meaning ema_signal and supertrend_signal were not yet both populated
        when each strategy was being scored. The gate always evaluated against
        stale "HOLD" values.

        Now: all strategies run first, results are collected, THEN the
        combined gate is evaluated once with all signals available.

    FIX (SELL RSI gate):
        Old condition: rsi < 50 or rsi > 65  (rsi > 65 is bullish — wrong for SELL)
        New condition: rsi < RSI_SELL_MAX     (RSI must show bearish momentum)

    FIX (threshold):
        Signal threshold raised from 25 to SIGNAL_THRESHOLD (40) to reduce
        false positives.
    """
    strategies = {
        "ema_crossover":      strategy_ema_crossover,
        "supertrend":         strategy_supertrend,
        "rsi_confirmation":   strategy_rsi_confirmation,
    }

    results = {}

    # --- PASS 1: Run all strategies and collect results ---
    for name, func in strategies.items():
        try:
            results[name] = func(df)
        except Exception as e:
            print(f"[Strategy] {name} failed: {e}")
            results[name] = {"signal": "HOLD", "confidence": 0.0, "error": str(e)}

    # --- Extract individual signals and RSI value ---
    ema_signal        = results.get("ema_crossover",    {}).get("signal", "HOLD")
    supertrend_signal = results.get("supertrend",       {}).get("signal", "HOLD")
    rsi_signal        = results.get("rsi_confirmation", {}).get("signal", "HOLD")
    rsi_value         = results.get("rsi_confirmation", {}).get("rsi", 50.0)

    ema_conf          = results.get("ema_crossover",    {}).get("confidence", 0.0)
    supertrend_conf   = results.get("supertrend",       {}).get("confidence", 0.0)
    rsi_conf          = results.get("rsi_confirmation", {}).get("confidence", 0.0)

    # --- PASS 2: Gate and score AFTER all strategies have run ---
    buy_score  = 0.0
    sell_score = 0.0

    # BUY gate: EMA bullish + SuperTrend bullish + RSI in building-momentum zone
    buy_gate = (
        ema_signal        == "BUY"
        and supertrend_signal == "BUY"
        and RSI_BUY_MIN <= rsi_value <= RSI_BUY_MAX
    )

    # SELL gate: EMA bearish + SuperTrend bearish + RSI showing bearish momentum
    # FIX: removed rsi > 65 from SELL condition — that is a bullish RSI zone
    sell_gate = (
        ema_signal        == "SELL"
        and supertrend_signal == "SELL"
        and rsi_value <= RSI_SELL_MAX
    )

    if buy_gate:
        buy_score = ema_conf + supertrend_conf + rsi_conf

    if sell_gate:
        sell_score = ema_conf + supertrend_conf + rsi_conf

    # Normalize: max possible = 3 strategies * 100 each
    max_possible  = len(strategies) * 100
    combined_buy  = round(buy_score  / max_possible * 100, 2)
    combined_sell = round(sell_score / max_possible * 100, 2)

    # FIX: Raised threshold from 25 to SIGNAL_THRESHOLD (40)
    if combined_buy >= SIGNAL_THRESHOLD:
        combined_signal     = "BUY"
        combined_confidence = combined_buy
    elif combined_sell >= SIGNAL_THRESHOLD:
        combined_signal     = "SELL"
        combined_confidence = combined_sell
    else:
        combined_signal     = "HOLD"
        combined_confidence = 0.0

    print(
        f"[Strategy] EMA={ema_signal}({ema_conf}%) | "
        f"ST={supertrend_signal}({supertrend_conf}%) | "
        f"RSI={rsi_value:.1f}({rsi_signal}) | "
        f"BuyGate={'PASS' if buy_gate else 'FAIL'} | "
        f"SellGate={'PASS' if sell_gate else 'FAIL'} | "
        f"Signal={combined_signal}({combined_confidence}%)"
    )

    return {
        "signal":     combined_signal,
        "confidence": combined_confidence,
        "buy_score":  combined_buy,
        "sell_score": combined_sell,
        "individual": results,
    }


# -----------------------------------------------
# ATR-BASED SL/TP CALCULATOR
# -----------------------------------------------

def calculate_sl_tp(entry: float, atr: float, side: str) -> Dict[str, float]:
    sl_dist  = ATR_SL_MULTIPLIER * atr
    tp1_dist = TP1_MULT * atr
    tp2_dist = TP2_MULT * atr
    tp3_dist = TP3_MULT * atr
    if side == "buy":
        return {
            "sl":  round(entry - sl_dist,  4),
            "tp1": round(entry + tp1_dist, 4),
            "tp2": round(entry + tp2_dist, 4),
            "tp3": round(entry + tp3_dist, 4),
        }
    else:
        return {
            "sl":  round(entry + sl_dist,  4),
            "tp1": round(entry - tp1_dist, 4),
            "tp2": round(entry - tp2_dist, 4),
            "tp3": round(entry - tp3_dist, 4),
        }


# -----------------------------------------------
# PYRAMID ADD-ON TRIGGER CHECKER
# -----------------------------------------------

def check_pyramid_trigger(
    entry: float,
    current_price: float,
    atr: float,
    side: str,
    pyramid_level: int,
) -> bool:
    required_move = pyramid_level * PYRAMID_ATR_TRIGGER * atr
    if side == "buy":
        return current_price >= entry + required_move
    else:
        return current_price <= entry - required_move


# -----------------------------------------------
# MAIN SIGNAL ENTRY POINT
# -----------------------------------------------

def generate_signal(
    data,
    mtf_data: Optional[Dict[str, list]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Main signal generator. Runs all strategies on primary timeframe
    and optionally incorporates multi-timeframe analysis.
    """
    try:
        df      = convert_to_dataframe(data)
        atr     = calculate_atr(df)
        combined = run_all_strategies(df)

        # Multi-timeframe analysis
        mtf_result = None
        if mtf_data:
            mtf_signals = {}
            for tf, candles in mtf_data.items():
                if candles:
                    try:
                        tf_df  = convert_to_dataframe(candles)
                        tf_res = strategy_ema_crossover(tf_df)
                        mtf_signals[tf] = tf_res
                    except Exception as e:
                        print(f"[Strategy] MTF {tf} failed: {e}")
            if mtf_signals:
                mtf_result = aggregate_mtf_signals(mtf_signals)

        final_signal     = combined["signal"]
        final_confidence = combined["confidence"]

        if mtf_result:
            mtf_sig  = mtf_result["signal"]
            mtf_conf = mtf_result["confidence"]
            if mtf_sig == final_signal:
                final_confidence = round(min((final_confidence * 0.6 + mtf_conf * 0.4), 100), 2)
            elif mtf_sig == "HOLD":
                final_confidence = round(final_confidence * 0.85, 2)
            else:
                final_signal     = "HOLD"
                final_confidence = 0.0

        sl_tp   = None
        entry   = float(df["Close"].iloc[-1])
        atr_val = float(atr.iloc[-1])

        if final_signal in ("BUY", "SELL"):
            side  = "buy" if final_signal == "BUY" else "sell"
            sl_tp = calculate_sl_tp(entry, atr_val, side)

        result = {
            "signal":            final_signal,
            "confidence":        final_confidence,
            "buy_score":         combined["buy_score"],
            "sell_score":        combined["sell_score"],
            "entry_price":       round(entry, 4),
            "atr":               round(atr_val, 4),
            "sl_tp":             sl_tp,
            "pyramid_trigger_1": round(PYRAMID_ATR_TRIGGER * atr_val, 4),
            "pyramid_trigger_2": round(2 * PYRAMID_ATR_TRIGGER * atr_val, 4),
            "tp_exit_pcts":      {"tp1": TP1_PCT, "tp2": TP2_PCT, "tp3": TP3_PCT},
            "individual":        combined["individual"],
            "mtf":               mtf_result,
            "rsi":               combined["individual"].get("rsi_confirmation", {}).get("rsi"),
            "ema_short":         combined["individual"].get("ema_crossover", {}).get("ema_short"),
            "ema_long":          combined["individual"].get("ema_crossover", {}).get("ema_long"),
            "supertrend":        combined["individual"].get("supertrend", {}).get("supertrend"),
        }

        print(
            f"[Strategy] Final Signal={final_signal} | "
            f"Confidence={final_confidence}% | "
            f"BuyScore={combined['buy_score']} | "
            f"SellScore={combined['sell_score']}"
        )

        return result

    except Exception as e:
        print(f"[Strategy] Critical error in generate_signal: {e}")
        return None
