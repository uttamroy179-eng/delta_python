# engine/executor.py

import httpx
import hashlib
import hmac
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# -----------------------------------------------
# CREDENTIALS & CONFIG
# -----------------------------------------------

project_root = Path(__file__).parent.parent
env_path     = project_root / ".env"

if not env_path.exists():
    raise FileNotFoundError(
        f".env file not found at: {env_path}\n"
        f"Please create a .env file in the project root with:\n"
        f"API_KEY=your_api_key\n"
        f"API_SECRET=your_api_secret"
    )

load_dotenv(env_path)

API_KEY    = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL   = "https://api.india.delta.exchange"

if not API_KEY or not API_SECRET:
    raise ValueError(
        f"Credentials not loaded from {env_path}.\n"
        f"API_KEY: {'set' if API_KEY else 'missing'}, "
        f"API_SECRET: {'set' if API_SECRET else 'missing'}"
    )

# Scale-out percentages per TP level
TP1_PCT = 0.20
TP2_PCT = 0.30
TP3_PCT = 0.50


# -----------------------------------------------
# SIGNATURE HELPER
# -----------------------------------------------

def generate_signature(secret: str, message: str) -> str:
    return hmac.new(
        bytes(secret, "utf-8"),
        bytes(message, "utf-8"),
        hashlib.sha256
    ).hexdigest()


def get_auth_headers(method: str, path: str,
                     query_string: str = "", payload: str = "") -> Dict[str, str]:
    timestamp      = str(int(time.time()))
    signature_data = method + timestamp + path + query_string + payload
    signature      = generate_signature(API_SECRET, signature_data)
    return {
        "api-key":      API_KEY,
        "timestamp":    timestamp,
        "signature":    signature,
        "Content-Type": "application/json",
        "User-Agent":   "autobot-trading-backend",
    }


# -----------------------------------------------
# PLACE SINGLE ORDER
# -----------------------------------------------

async def place_order(symbol: str, side: str, size: int,
                      order_type: str = "market_order",
                      limit_price: Optional[float] = None) -> Optional[Dict]:
    """
    Place a single order on Delta Exchange.

    Args:
        symbol      : Trading symbol e.g. BTCUSD
        side        : "buy" or "sell"
        size        : Number of contracts (integer, no fractional lots)
        order_type  : "market_order" or "limit_order"
        limit_price : Required if order_type is "limit_order"

    Returns:
        API response dict or None on failure.
    """
    try:
        path = "/v2/orders"
        body: Dict[str, Any] = {
            "product_symbol": symbol,
            "side":           side,
            "size":           size,
            "order_type":     order_type,
        }

        if order_type == "limit_order" and limit_price is not None:
            body["limit_price"] = str(limit_price)

        payload   = json.dumps(body)
        headers   = get_auth_headers("POST", path, payload=payload)

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{BASE_URL}{path}",
                content=payload,
                headers=headers
            )

        result = response.json()

        if result.get("success"):
            order_id = result.get("result", {}).get("id", "N/A")
            print(f"[Executor] Order placed: {side.upper()} {size} {symbol} "
                  f"({order_type}) | Order ID: {order_id}")
        else:
            print(f"[Executor] Order failed for {symbol}: {result.get('error', 'Unknown error')}")

        return result

    except Exception as e:
        print(f"[Executor] Exception placing order for {symbol}: {e}")
        return None


# -----------------------------------------------
# PYRAMID IN — PLACE ADD-ON ENTRY
# -----------------------------------------------

async def place_pyramid_order(symbol: str, side: str,
                               base_size: int, level: int) -> Optional[Dict]:
    """
    Place a pyramid add-on order. Size is same as base_size for each level.

    Args:
        symbol    : Trading symbol
        side      : "buy" or "sell"
        base_size : Size of the initial order
        level     : Pyramid level (1 or 2)

    Returns:
        API response dict or None on failure.
    """
    print(f"[Executor] Pyramid Level {level} add-on: {side.upper()} {base_size} {symbol}")
    return await place_order(symbol, side, base_size)


# -----------------------------------------------
# SCALE OUT — PARTIAL EXIT AT TP LEVEL
# -----------------------------------------------

async def scale_out(symbol: str, side: str,
                    total_size: int, tp_level: int) -> Optional[Dict]:
    """
    Close a portion of the position at a given TP level.

    TP1 -> close 20% of position
    TP2 -> close 30% of position
    TP3 -> close 50% of position

    Args:
        symbol     : Trading symbol
        side       : Original position side ("buy" -> close with "sell")
        total_size : Total open position size
        tp_level   : 1, 2, or 3

    Returns:
        API response dict or None on failure.
    """
    pct_map = {1: TP1_PCT, 2: TP2_PCT, 3: TP3_PCT}
    pct     = pct_map.get(tp_level, 0)

    # Calculate close size — minimum 1 contract, no fractional lots
    close_size = max(1, round(total_size * pct))

    # Closing side is opposite of entry side
    close_side = "sell" if side == "buy" else "buy"

    print(f"[Executor] Scale Out TP{tp_level}: {close_side.upper()} {close_size} "
          f"of {total_size} {symbol} ({int(pct * 100)}%)")

    return await place_order(symbol, close_side, close_size)


# -----------------------------------------------
# CLOSE FULL POSITION (SL HIT OR DAILY LOSS LIMIT)
# -----------------------------------------------

async def close_position(symbol: str, side: str, size: int) -> Optional[Dict]:
    """
    Close the entire remaining position (SL hit or forced close).

    Args:
        symbol : Trading symbol
        side   : Original position side
        size   : Remaining open size to close
    """
    close_side = "sell" if side == "buy" else "buy"
    print(f"[Executor] Closing full position: {close_side.upper()} {size} {symbol}")
    return await place_order(symbol, close_side, size)


# -----------------------------------------------
# FETCH OPEN POSITIONS
# -----------------------------------------------

async def fetch_open_positions() -> list:
    """
    Fetch all open positions from Delta Exchange.

    Returns:
        List of position dicts or empty list on failure.
    """
    try:
        path    = "/v2/positions/margined"
        headers = get_auth_headers("GET", path)

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{BASE_URL}{path}",
                headers=headers
            )

        result = response.json()

        if result.get("success"):
            return result.get("result", [])
        else:
            print(f"[Executor] Failed to fetch positions: {result.get('error')}")
            return []

    except Exception as e:
        print(f"[Executor] Exception fetching positions: {e}")
        return []


# -----------------------------------------------
# FETCH WALLET BALANCE (USDT)
# -----------------------------------------------

async def fetch_wallet_balance() -> Optional[float]:
    """
    Fetch available USDT wallet balance from Delta Exchange.

    Returns:
        Available USDT balance as float, or None on failure.
    """
    try:
        path    = "/v2/wallet/balances"
        headers = get_auth_headers("GET", path)

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{BASE_URL}{path}",
                headers=headers
            )

        data = response.json()
        print("[Executor] Wallet API Response:", data)

        if data.get("success"):
            for wallet in data.get("result", []):
                asset_symbol = (
                    wallet.get("asset_symbol")
                    or wallet.get("asset", {}).get("symbol")
                )

                if asset_symbol in ["USDT", "USD"]:
                    balance = float(
                        wallet.get("available_balance", 0)
                    )

                    print(f"[Executor] Available USDT Balance: {balance:.2f}")
                    return balance

            print("[Executor] Could not find USDT balance in wallet response")
            return None
        else:
            print(f"[Executor] Failed to fetch wallet balances: {data.get('error')}")
            return None

    except Exception as e:
        print(f"[Executor] Exception fetching wallet balance: {e}")
        return None
