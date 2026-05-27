# engine/data_fetcher.py

import httpx
import time

BASE_URL = "https://api.india.delta.exchange"

RESOLUTION_SECONDS = {
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "2h":  7200,
    "4h":  14400,
    "1d":  86400,
}


async def fetch_ohlc(symbol: str, resolution: str = "5m", limit: int = 100):
    """
    Fetch OHLC candle data from Delta Exchange.

    Args:
        symbol     : Trading symbol e.g. BTCUSD, ETHUSD
        resolution : Candle resolution - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d
        limit      : Number of candles to fetch (default 100)

    Returns:
        List of candle dicts with keys: time, open, high, low, close, volume
        Returns empty list on failure.
    """
    try:
        end_time   = int(time.time())
        interval   = RESOLUTION_SECONDS.get(resolution, 300)
        start_time = end_time - (interval * limit)

        url = f"{BASE_URL}/v2/history/candles"

        params = {
            "symbol":     symbol,
            "resolution": resolution,
            "start":      start_time,
            "end":        end_time,
        }

        headers = {
            "User-Agent": "autobot-trading-backend",
            "Accept":     "application/json",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)

        if response.status_code != 200:
            print(f"[DataFetcher] OHLC fetch failed for {symbol} ({resolution}) - HTTP {response.status_code}")
            return []

        response_json = response.json()

        if not response_json.get("success", False):
            print(f"[DataFetcher] Delta Exchange returned success=false for {symbol} ({resolution})")
            return []

        candles = response_json.get("result", [])

        if not candles:
            print(f"[DataFetcher] No candles returned for {symbol} ({resolution})")
            return []

        processed = []
        for candle in candles:
            processed.append({
                "time":   candle.get("time"),
                "open":   float(candle.get("open",   0)),
                "high":   float(candle.get("high",   0)),
                "low":    float(candle.get("low",    0)),
                "close":  float(candle.get("close",  0)),
                "volume": float(candle.get("volume", 0)),
            })

        # Ensure chronological order
        processed.sort(key=lambda x: x["time"])

        print(f"[DataFetcher] Fetched {len(processed)} candles for {symbol} ({resolution})")
        return processed

    except Exception as e:
        print(f"[DataFetcher] Error fetching OHLC for {symbol} ({resolution}): {e}")
        return []


async def fetch_ohlc_multi_timeframe(symbol: str, timeframes: list = None, limit: int = 100):
    """
    Fetch OHLC data for multiple timeframes for a given symbol.

    Args:
        symbol     : Trading symbol e.g. BTCUSD
        timeframes : List of resolution strings e.g. ["5m", "15m"]
        limit      : Number of candles per timeframe

    Returns:
        Dict keyed by timeframe: {"5m": [...], "15m": [...]}
    """
    if timeframes is None:
        timeframes = ["5m", "15m"]

    result = {}
    for tf in timeframes:
        candles = await fetch_ohlc(symbol, resolution=tf, limit=limit)
        result[tf] = candles

    return result
