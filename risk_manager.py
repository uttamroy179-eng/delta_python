# engine/risk_manager.py

import threading
from datetime import datetime, date
from typing import Dict, Optional, Any

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------

MAX_DAILY_LOSS_PCT  = 20.0   # Stop trading if daily loss exceeds 20% of starting balance
CAPITAL_RISK_PCT    = 2.0    # Risk 2% of balance per trade
ATR_SL_MULTIPLIER   = 1.5
MAX_CONTRACTS       = 50     # FIX #4: Configurable hard cap (replaces magic number)
MAX_PYRAMID_LEVELS  = 2      # FIX #10: Maximum allowed pyramid add-on levels

# Scale-out percentages
# FIX #12: These are the single source of truth.
# strategy.py must import from here instead of redefining them.
TP1_PCT = 0.20
TP2_PCT = 0.30
TP3_PCT = 0.50


# -----------------------------------------------
# DAILY LOSS TRACKER
# -----------------------------------------------

class DailyLossTracker:
    """
    Tracks realized PnL for the current trading day.
    Resets automatically at the start of each new day.

    Thread-safety: All public methods are protected by a reentrant lock.
    FIX #11: Added threading.RLock() to prevent race conditions in
             async/threaded environments.
    """

    def __init__(self, starting_balance: float):
        # FIX #11: RLock allows the same thread to re-acquire the lock
        # (needed because check_daily_loss_limit calls _check_date_reset internally)
        self._lock             = threading.RLock()

        self.starting_balance  = starting_balance
        self.realized_pnl      = 0.0
        self.trade_count       = 0
        self.last_reset_date   = date.today()
        self.trading_halted    = False
        self.halt_reason: Optional[str] = None

        # FIX #1: Track whether the balance has been refreshed for the new day
        self._balance_needs_refresh = False

    def _check_date_reset(self):
        """
        Reset tracker if a new trading day has started.
        FIX #1: Sets a flag so the caller knows to refresh starting_balance
                before the first trade of the new day.
        """
        today = date.today()
        if today != self.last_reset_date:
            print(
                f"[RiskManager] New trading day detected ({self.last_reset_date} -> {today}). "
                f"Resetting daily PnL tracker."
            )
            self.realized_pnl           = 0.0
            self.trade_count            = 0
            self.trading_halted         = False
            self.halt_reason            = None
            self.last_reset_date        = today
            # FIX #1: Flag that starting_balance must be updated before trading
            self._balance_needs_refresh = True

    def record_trade(self, pnl: float):
        """
        Record a completed trade's PnL.
        Raises RuntimeError if starting balance has not been refreshed
        for the new day yet.
        FIX #1: Guards against stale starting_balance on a new day.
        """
        with self._lock:
            self._check_date_reset()

            # FIX #1: Block trading until balance is refreshed for the new day
            if self._balance_needs_refresh:
                raise RuntimeError(
                    "[RiskManager] Starting balance must be updated via "
                    "update_starting_balance() before recording trades on a new day."
                )

            self.realized_pnl += pnl
            self.trade_count  += 1
            print(
                f"[RiskManager] Trade recorded. PnL: {pnl:.2f} USDT | "
                f"Daily PnL: {self.realized_pnl:.2f} USDT | "
                f"Trades today: {self.trade_count}"
            )

    def update_starting_balance(self, balance: float):
        """
        Update the starting balance reference.

        FIX #2: Removed the overly restrictive `realized_pnl == 0.0` guard.
                Balance is now updated whenever:
                  (a) A new day has been detected (_balance_needs_refresh is True), OR
                  (b) No trades have been recorded yet today (trade_count == 0).
                This prevents the silent no-op bug where a profitable trade
                would permanently block balance updates.
        """
        with self._lock:
            self._check_date_reset()

            if self._balance_needs_refresh or self.trade_count == 0:
                self.starting_balance       = balance
                self._balance_needs_refresh = False
                print(f"[RiskManager] Starting balance updated: {balance:.2f} USDT")
            else:
                print(
                    f"[RiskManager] Starting balance NOT updated: "
                    f"{self.trade_count} trade(s) already recorded today. "
                    f"Current starting balance: {self.starting_balance:.2f} USDT"
                )

    def check_daily_loss_limit(self) -> Dict[str, Any]:
        """
        Check if the daily loss limit has been breached.

        Returns:
            Dict with keys:
              - halted (bool)         : True if trading should stop
              - daily_pnl (float)     : Current day's realized PnL
              - daily_loss_pct (float): Loss as % of starting balance
              - limit_pct (float)     : Configured limit
              - trade_count (int)     : Number of trades today
              - starting_balance (float)
              - message (str)         : Human-readable status
        """
        with self._lock:
            self._check_date_reset()

            daily_loss_pct = 0.0
            if self.starting_balance > 0:
                daily_loss_pct = (self.realized_pnl / self.starting_balance) * 100

            limit_breached = daily_loss_pct <= -MAX_DAILY_LOSS_PCT

            if limit_breached and not self.trading_halted:
                self.trading_halted = True
                self.halt_reason    = (
                    f"Daily loss limit of {MAX_DAILY_LOSS_PCT}% breached. "
                    f"Current loss: {abs(daily_loss_pct):.2f}%"
                )
                print(f"[RiskManager] TRADING HALTED: {self.halt_reason}")

            return {
                "halted":           self.trading_halted,
                "daily_pnl":        round(self.realized_pnl, 2),
                "daily_loss_pct":   round(daily_loss_pct, 2),
                "limit_pct":        MAX_DAILY_LOSS_PCT,
                "trade_count":      self.trade_count,
                "starting_balance": round(self.starting_balance, 2),
                "message": (
                    self.halt_reason if self.trading_halted
                    else f"Trading active. Daily PnL: {self.realized_pnl:.2f} USDT "
                         f"({daily_loss_pct:.2f}%)"
                ),
            }

    def is_trading_allowed(self) -> bool:
        """Returns True if trading is allowed (daily loss limit not breached)."""
        status = self.check_daily_loss_limit()
        return not status["halted"]

    def resume_trading(self, reason: str = "Manual override"):
        """
        FIX #3: Manually resume trading after a halt.
        Logs an audit entry with timestamp and reason for the override.
        Use with caution — only call after human review.
        """
        with self._lock:
            if not self.trading_halted:
                print("[RiskManager] resume_trading called but trading is not halted. No action taken.")
                return

            audit_entry = (
                f"[RiskManager] TRADING RESUMED at {datetime.now().isoformat()} | "
                f"Reason: {reason} | "
                f"Daily PnL at resume: {self.realized_pnl:.2f} USDT | "
                f"Loss pct: "
                f"{abs((self.realized_pnl / self.starting_balance) * 100):.2f}% "
                f"(limit: {MAX_DAILY_LOSS_PCT}%)"
            )
            print(audit_entry)

            self.trading_halted = False
            self.halt_reason    = None


# -----------------------------------------------
# POSITION SIZER
# -----------------------------------------------

def calculate_position_size(
    balance: float,
    atr: float,
    entry_price: float,
    leverage: int = 5,
    contract_value: float = 1.0,
    max_balance_usage: float = 0.50,
    min_sl_pct: float = 0.002,
    max_contracts: int = MAX_CONTRACTS,   # FIX #4: Configurable cap
) -> int:
    """
    Safe position sizing with:
    - Input validation                    (FIX #5)
    - ATR stop loss
    - Leverage protection
    - Margin protection
    - Minimum SL distance
    - Configurable contract cap           (FIX #4)

    Args:
        balance          : Current account balance in USDT
        atr              : Current ATR value for the symbol
        entry_price      : Intended entry price
        leverage         : Leverage to apply (default 5)
        contract_value   : Notional value per contract from Delta Exchange
                           product specs (contract_value field). Default 1.0.
                           For BTCUSD this is typically 0.001 BTC per contract.
        max_balance_usage: Max fraction of balance to use as margin (default 50%)
        min_sl_pct       : Minimum SL distance as % of entry price (default 0.2%)
        max_contracts    : Hard cap on number of contracts (default MAX_CONTRACTS)

    Returns:
        Integer number of contracts (minimum 1).
    """

    # FIX #5: Validate all primary inputs before any calculation
    if balance <= 0:
        print(f"[RiskManager] Invalid balance: {balance}. Returning minimum size 1.")
        return 1
    if entry_price <= 0:
        print(f"[RiskManager] Invalid entry_price: {entry_price}. Returning minimum size 1.")
        return 1
    if leverage <= 0:
        print(f"[RiskManager] Invalid leverage: {leverage}. Returning minimum size 1.")
        return 1
    if contract_value <= 0:
        print(f"[RiskManager] Invalid contract_value: {contract_value}. Returning minimum size 1.")
        return 1
    if atr <= 0:
        print(f"[RiskManager] Invalid ATR: {atr}. Returning minimum size 1.")
        return 1

    # Risk amount in USDT
    risk_amount = balance * (CAPITAL_RISK_PCT / 100)

    # ATR-based SL distance
    sl_distance = ATR_SL_MULTIPLIER * atr

    # Prevent microscopic SL
    min_sl_distance = entry_price * min_sl_pct
    sl_distance     = max(sl_distance, min_sl_distance)

    # Risk-based size
    # contract_value converts contracts -> underlying units
    # sl_distance * contract_value = USDT risk per contract (vanilla contracts)
    risk_size = risk_amount / (sl_distance * contract_value)

    # Maximum position allowed by margin
    usable_balance      = balance * max_balance_usage
    max_position_value  = usable_balance * leverage
    max_size            = max_position_value / entry_price

    # Final safe size
    size = min(risk_size, max_size)

    # FIX #4: Apply configurable hard cap and log a warning if triggered
    if size > max_contracts:
        print(
            f"[RiskManager] WARNING: Calculated size {int(size)} exceeds "
            f"max_contracts cap of {max_contracts}. Capping at {max_contracts}."
        )
        size = max_contracts

    # Integer contracts only (Delta Exchange does not support fractional lots)
    size = max(1, int(size))

    required_margin = (size * entry_price * contract_value) / leverage

    print(
        f"[RiskManager] Position size: {size} contracts | "
        f"Risk: {risk_amount:.2f} USDT | "
        f"SL distance: {sl_distance:.4f} | "
        f"Contract value: {contract_value} | "
        f"Required Margin: {required_margin:.2f} USDT"
    )

    return size


# -----------------------------------------------
# POSITION STATE TRACKER
# -----------------------------------------------

class PositionState:
    """
    Tracks the state of an open position including
    pyramid levels and TP hit status.

    FIX #9 : contract_value is now stored and used in all PnL calculations.
    FIX #10: Pyramid level is capped at MAX_PYRAMID_LEVELS.
    FIX #11: All state mutations are protected by a threading.RLock.
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        base_size: int,
        atr: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        contract_value: float = 1.0,   # FIX #9: Accept contract_value at construction
    ):
        # FIX #11: Lock for thread-safe state mutations
        self._lock = threading.RLock()

        self.symbol         = symbol
        self.side           = side
        self.entry_price    = entry_price
        self.base_size      = base_size
        self.current_size   = base_size
        self.atr            = atr
        self.sl             = sl
        self.tp1            = tp1
        self.tp2            = tp2
        self.tp3            = tp3
        self.pyramid_level  = 0
        self.tp1_hit        = False
        self.tp2_hit        = False
        self.tp3_hit        = False
        self.entry_time     = datetime.now().isoformat()
        self.realized_pnl   = 0.0

        # FIX #9: Store contract_value for correct PnL calculation
        # Sourced from Delta Exchange product specs: result.contract_value
        self.contract_value = contract_value

    def add_pyramid(self, size: int):
        """
        Record a pyramid add-on.
        FIX #10: Raises ValueError if MAX_PYRAMID_LEVELS is already reached.
        """
        with self._lock:
            # FIX #10: Enforce maximum pyramid level
            if self.pyramid_level >= MAX_PYRAMID_LEVELS:
                raise ValueError(
                    f"[RiskManager] Cannot add pyramid for {self.symbol}: "
                    f"already at maximum level {MAX_PYRAMID_LEVELS}."
                )

            self.pyramid_level += 1
            self.current_size  += size
            print(
                f"[RiskManager] Pyramid level {self.pyramid_level} added. "
                f"New size: {self.current_size} for {self.symbol}"
            )

    def record_tp_exit(self, tp_level: int, exit_price: float, closed_size: int) -> float:
        """
        Record a partial TP exit and update realized PnL.

        FIX #6: closed_size is clamped to current_size to prevent
                over-closing and incorrect PnL.
        FIX #7: Guard against duplicate TP hits at the same level.
        FIX #9: PnL multiplied by contract_value for correct USDT value.

        Returns:
            Realized PnL for this exit in USDT.
        """
        with self._lock:
            # FIX #7: Prevent duplicate TP hits
            tp_hit_flags = {1: self.tp1_hit, 2: self.tp2_hit, 3: self.tp3_hit}
            if tp_level not in tp_hit_flags:
                raise ValueError(f"[RiskManager] Invalid tp_level: {tp_level}. Must be 1, 2, or 3.")

            if tp_hit_flags[tp_level]:
                print(
                    f"[RiskManager] WARNING: TP{tp_level} already hit for {self.symbol}. "
                    f"Ignoring duplicate exit call."
                )
                return 0.0

            # FIX #6: Clamp closed_size to what is actually available
            if closed_size > self.current_size:
                print(
                    f"[RiskManager] WARNING: closed_size {closed_size} exceeds "
                    f"current_size {self.current_size} for {self.symbol}. "
                    f"Clamping to {self.current_size}."
                )
                closed_size = self.current_size

            if closed_size <= 0:
                print(f"[RiskManager] WARNING: closed_size is 0 for {self.symbol}. No exit recorded.")
                return 0.0

            # FIX #9: Multiply by contract_value to get correct USDT PnL
            if self.side == "buy":
                pnl = (exit_price - self.entry_price) * closed_size * self.contract_value
            else:
                pnl = (self.entry_price - exit_price) * closed_size * self.contract_value

            self.realized_pnl += pnl
            self.current_size  = max(0, self.current_size - closed_size)

            if tp_level == 1:
                self.tp1_hit = True
            elif tp_level == 2:
                self.tp2_hit = True
            elif tp_level == 3:
                self.tp3_hit = True

            print(
                f"[RiskManager] TP{tp_level} hit for {self.symbol}. "
                f"Closed {closed_size} contracts at {exit_price}. "
                f"PnL: {pnl:.2f} USDT. "
                f"Remaining size: {self.current_size}"
            )

            return pnl

    def record_sl_exit(self, exit_price: float) -> float:
        """
        Record a full SL exit and return total PnL.

        FIX #8: Guard against calling SL exit on an already-closed position.
        FIX #9: PnL multiplied by contract_value for correct USDT value.

        Returns:
            Realized PnL for this exit in USDT, or 0.0 if already closed.
        """
        with self._lock:
            # FIX #8: Guard against SL exit on a fully closed position
            if self.current_size == 0:
                print(
                    f"[RiskManager] WARNING: record_sl_exit called for {self.symbol} "
                    f"but position is already fully closed. Ignoring."
                )
                return 0.0

            # FIX #9: Multiply by contract_value to get correct USDT PnL
            if self.side == "buy":
                pnl = (exit_price - self.entry_price) * self.current_size * self.contract_value
            else:
                pnl = (self.entry_price - exit_price) * self.current_size * self.contract_value

            self.realized_pnl += pnl
            closed_size        = self.current_size
            self.current_size  = 0

            print(
                f"[RiskManager] SL hit for {self.symbol}. "
                f"Closed {closed_size} contracts at {exit_price}. "
                f"PnL: {pnl:.2f} USDT"
            )

            return pnl

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "symbol":         self.symbol,
                "side":           self.side,
                "entry_price":    self.entry_price,
                "base_size":      self.base_size,
                "current_size":   self.current_size,
                "atr":            self.atr,
                "sl":             self.sl,
                "tp1":            self.tp1,
                "tp2":            self.tp2,
                "tp3":            self.tp3,
                "pyramid_level":  self.pyramid_level,
                "tp1_hit":        self.tp1_hit,
                "tp2_hit":        self.tp2_hit,
                "tp3_hit":        self.tp3_hit,
                "entry_time":     self.entry_time,
                "realized_pnl":   round(self.realized_pnl, 2),
                "contract_value": self.contract_value,
            }
