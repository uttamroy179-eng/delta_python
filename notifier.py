# engine/notifier.py

import httpx
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from datetime import datetime

# -----------------------------------------------
# LOAD CREDENTIALS
# -----------------------------------------------

project_root = Path(__file__).parent.parent
env_path     = project_root / ".env"
load_dotenv(env_path)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

TELEGRAM_API_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    if TELEGRAM_BOT_TOKEN else None
)


# -----------------------------------------------
# LOW-LEVEL SEND
# -----------------------------------------------

async def _send_telegram(message: str) -> bool:
    """
    Send a plain text message to the configured Telegram chat.

    Returns True on success, False on failure.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Notifier] Telegram credentials not configured. Skipping notification.")
        return False

    try:
        payload = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(TELEGRAM_API_URL, json=payload)

        if response.status_code == 200:
            print("[Notifier] Telegram message sent successfully.")
            return True
        else:
            print(f"[Notifier] Telegram send failed: HTTP {response.status_code} | {response.text}")
            return False

    except Exception as e:
        print(f"[Notifier] Exception sending Telegram message: {e}")
        return False


# -----------------------------------------------
# SIGNAL ALERT
# -----------------------------------------------

async def notify_signal(symbol: str, signal: str, confidence: float,
                         entry_price: float, atr: float,
                         sl_tp: Optional[Dict[str, float]],
                         daily_loss_status: Dict[str, Any]) -> bool:
    """
    Send a BUY/SELL signal alert to Telegram.

    Includes:
      - Symbol, signal type, confidence score
      - Entry price, SL, TP1/TP2/TP3
      - Daily PnL and loss limit status
    """
    if signal not in ("BUY", "SELL"):
        return False

    emoji = "BUY" if signal == "BUY" else "SELL"
    direction_arrow = "UP" if signal == "BUY" else "DOWN"

    sl_tp_section = ""
    if sl_tp:
        sl_tp_section = (
            f"\n"
            f"<b>SL:</b>  {sl_tp.get('sl', 'N/A')}\n"
            f"<b>TP1:</b> {sl_tp.get('tp1', 'N/A')} (close 20%)\n"
            f"<b>TP2:</b> {sl_tp.get('tp2', 'N/A')} (close 30%)\n"
            f"<b>TP3:</b> {sl_tp.get('tp3', 'N/A')} (close 50%)"
        )

    daily_pnl     = daily_loss_status.get("daily_pnl", 0.0)
    daily_pnl_pct = daily_loss_status.get("daily_loss_pct", 0.0)
    halted        = daily_loss_status.get("halted", False)
    trade_count   = daily_loss_status.get("trade_count", 0)

    daily_status_line = (
        f"HALTED ({abs(daily_pnl_pct):.2f}% loss)"
        if halted
        else f"{daily_pnl:+.2f} USDT ({daily_pnl_pct:+.2f}%)"
    )

    message = (
        f"<b>[{emoji}] {signal} Signal — {symbol}</b>\n"
        f"{'=' * 30}\n"
        f"<b>Direction:</b>  {direction_arrow}\n"
        f"<b>Confidence:</b> {confidence:.1f}%\n"
        f"<b>Entry Price:</b> {entry_price}\n"
        f"<b>ATR:</b>        {atr:.4f}"
        f"{sl_tp_section}\n"
        f"{'=' * 30}\n"
        f"<b>Daily PnL:</b>    {daily_status_line}\n"
        f"<b>Trades Today:</b> {trade_count}\n"
        f"<b>Time:</b>         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    return await _send_telegram(message)


# -----------------------------------------------
# DAILY LOSS LIMIT ALERT
# -----------------------------------------------

async def notify_daily_loss_limit(symbol: str,
                                   daily_loss_status: Dict[str, Any]) -> bool:
    """
    Send an alert when the daily loss limit is breached
    and trading has been halted.
    """
    daily_pnl     = daily_loss_status.get("daily_pnl", 0.0)
    daily_pnl_pct = daily_loss_status.get("daily_loss_pct", 0.0)
    limit_pct     = daily_loss_status.get("limit_pct", 20.0)
    trade_count   = daily_loss_status.get("trade_count", 0)
    start_bal     = daily_loss_status.get("starting_balance", 0.0)

    message = (
        f"<b>[ALERT] Daily Loss Limit Breached</b>\n"
        f"{'=' * 30}\n"
        f"<b>Triggered by:</b>  {symbol}\n"
        f"<b>Daily Loss:</b>    {abs(daily_pnl):.2f} USDT ({abs(daily_pnl_pct):.2f}%)\n"
        f"<b>Limit:</b>         {limit_pct:.0f}%\n"
        f"<b>Start Balance:</b> {start_bal:.2f} USDT\n"
        f"<b>Trades Today:</b>  {trade_count}\n"
        f"<b>Status:</b>        TRADING HALTED for today\n"
        f"<b>Time:</b>          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 30}\n"
        f"Bot will resume trading tomorrow."
    )

    return await _send_telegram(message)


# -----------------------------------------------
# TP HIT ALERT
# -----------------------------------------------

async def notify_tp_hit(symbol: str, tp_level: int,
                         exit_price: float, closed_size: int,
                         pnl: float, remaining_size: int,
                         daily_loss_status: Dict[str, Any]) -> bool:
    """
    Send a notification when a TP level is hit and partial exit is executed.
    """
    pct_map = {1: "20%", 2: "30%", 3: "50%"}
    pct_closed = pct_map.get(tp_level, "N/A")

    daily_pnl     = daily_loss_status.get("daily_pnl", 0.0)
    daily_pnl_pct = daily_loss_status.get("daily_loss_pct", 0.0)

    message = (
        f"<b>[TP{tp_level} HIT] {symbol}</b>\n"
        f"{'=' * 30}\n"
        f"<b>Exit Price:</b>    {exit_price}\n"
        f"<b>Closed:</b>        {closed_size} contracts ({pct_closed})\n"
        f"<b>Trade PnL:</b>     {pnl:+.2f} USDT\n"
        f"<b>Remaining:</b>     {remaining_size} contracts\n"
        f"{'=' * 30}\n"
        f"<b>Daily PnL:</b>     {daily_pnl:+.2f} USDT ({daily_pnl_pct:+.2f}%)\n"
        f"<b>Time:</b>          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    return await _send_telegram(message)


# -----------------------------------------------
# SL HIT ALERT
# -----------------------------------------------

async def notify_sl_hit(symbol: str, exit_price: float,
                         closed_size: int, pnl: float,
                         daily_loss_status: Dict[str, Any]) -> bool:
    """
    Send a notification when SL is hit and full position is closed.
    """
    daily_pnl     = daily_loss_status.get("daily_pnl", 0.0)
    daily_pnl_pct = daily_loss_status.get("daily_loss_pct", 0.0)
    halted        = daily_loss_status.get("halted", False)

    halt_line = "\nTRADING HALTED for today (daily loss limit reached)." if halted else ""

    message = (
        f"<b>[SL HIT] {symbol}</b>\n"
        f"{'=' * 30}\n"
        f"<b>Exit Price:</b>  {exit_price}\n"
        f"<b>Closed:</b>      {closed_size} contracts (full position)\n"
        f"<b>Trade PnL:</b>   {pnl:+.2f} USDT\n"
        f"{'=' * 30}\n"
        f"<b>Daily PnL:</b>   {daily_pnl:+.2f} USDT ({daily_pnl_pct:+.2f}%)\n"
        f"<b>Time:</b>        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"{halt_line}"
    )

    return await _send_telegram(message)


# -----------------------------------------------
# PYRAMID ADD-ON ALERT
# -----------------------------------------------

async def notify_pyramid(symbol: str, level: int, side: str,
                          size: int, trigger_price: float,
                          daily_loss_status: Dict[str, Any]) -> bool:
    """
    Send a notification when a pyramid add-on order is placed.
    """
    daily_pnl = daily_loss_status.get("daily_pnl", 0.0)

    message = (
        f"<b>[PYRAMID L{level}] {symbol}</b>\n"
        f"{'=' * 30}\n"
        f"<b>Side:</b>          {side.upper()}\n"
        f"<b>Add-on Size:</b>   {size} contracts\n"
        f"<b>Trigger Price:</b> {trigger_price}\n"
        f"{'=' * 30}\n"
        f"<b>Daily PnL:</b>     {daily_pnl:+.2f} USDT\n"
        f"<b>Time:</b>          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    return await _send_telegram(message)
