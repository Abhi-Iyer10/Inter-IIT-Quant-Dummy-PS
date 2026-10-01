"""
strategy_base.py - Core API Contracts for Quantitative Strategies
Inter IIT Tech Meet 15.0 Quant Evaluation Architecture
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Mapping


class Bar:
    """
    Immutable market bar snapshot at time step t.
    Supports both Track A (Daily Cross-Sectional OHLCV) and
    Track B (1-second Microstructure OHLCV + Order Flow Imbalance).
    """
    __slots__ = (
        'ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'count', 'taker_buy_volume',
        '_ofi', '_vwap', '_log_range'
    )

    def __init__(
        self,
        ticker: str,
        timestamp: Any,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        quote_volume: float = 0.0,
        count: int = 0,
        taker_buy_volume: float = 0.0,
    ) -> None:
        self.ticker = ticker
        self.timestamp = timestamp
        self.open = float(open)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = float(volume)
        self.quote_volume = float(quote_volume)
        self.count = int(count)
        self.taker_buy_volume = float(taker_buy_volume)
        self._ofi: Optional[float] = None
        self._vwap: Optional[float] = None
        self._log_range: Optional[float] = None

    @property
    def ofi(self) -> float:
        """
        Aggressive Order Flow Imbalance (OFI) in range [-1.0, 1.0].
        OFI = (taker_buy_volume - taker_sell_volume) / volume
        """
        if self._ofi is None:
            if self.volume > 0.0:
                taker_sell = self.volume - self.taker_buy_volume
                val = (self.taker_buy_volume - taker_sell) / self.volume
                self._ofi = max(-1.0, min(1.0, val))
            else:
                self._ofi = 0.0
        return self._ofi

    @property
    def vwap(self) -> float:
        """
        Volume-Weighted Average Price for the bar.
        """
        if self._vwap is None:
            if self.volume > 0.0 and self.quote_volume > 0.0:
                self._vwap = self.quote_volume / self.volume
            else:
                self._vwap = self.close
        return self._vwap

    @property
    def log_range(self) -> float:
        """
        Intraday Volatility Proxy (Log Range): ln(high) - ln(low).
        """
        if self._log_range is None:
            if self.high > 0.0 and self.low > 0.0 and self.high >= self.low:
                self._log_range = math.log(self.high) - math.log(self.low)
            else:
                self._log_range = 0.0
        return self._log_range

    @property
    def taker_sell_volume(self) -> float:
        """
        Passive buyer / aggressive seller volume: volume - taker_buy_volume.
        """
        return max(0.0, self.volume - self.taker_buy_volume)

    def __repr__(self) -> str:
        return (
            f"Bar(ticker={self.ticker!r}, timestamp={self.timestamp!r}, "
            f"O={self.open:.2f}, H={self.high:.2f}, L={self.low:.2f}, C={self.close:.2f}, V={self.volume:.1f})"
        )


class FastBarView(Bar):
    """
    Mutable flyweight bar view used for ultra-high-throughput streaming (Track B).
    Reuses memory buffers to prevent 3.88M object allocations in hot loops.
    """
    __slots__ = ()

    def update(
        self,
        ticker: str,
        timestamp: Any,
        open_val: float,
        high_val: float,
        low_val: float,
        close_val: float,
        vol_val: float,
        qvol_val: float,
        count_val: int,
        tbv_val: float,
    ) -> None:
        self.ticker = ticker
        self.timestamp = timestamp
        self.open = open_val
        self.high = high_val
        self.low = low_val
        self.close = close_val
        self.volume = vol_val
        self.quote_volume = qvol_val
        self.count = count_val
        self.taker_buy_volume = tbv_val
        self._ofi = None
        self._vwap = None
        self._log_range = None


class Context:
    """
    Evaluation Context passed into strategy callbacks.
    Provides portfolio status and collects participant target allocations.
    """

    def __init__(
        self,
        initial_cash: float,
        universe: Optional[List[str]] = None,
        symbol: Optional[str] = None
    ) -> None:
        self.initial_cash: float = float(initial_cash)
        self.cash: float = float(initial_cash)
        self.portfolio_value: float = float(initial_cash)
        self.step: int = 0
        self.timestamp: Any = None
        self.universe: List[str] = list(universe or [])
        self.symbol: Optional[str] = symbol
        
        # Read-only internal state exposed to strategy
        self._positions: Dict[str, float] = {}
        self._target_weights: Dict[str, float] = {}
        
        # Custom user scratchpad storage
        self.extra: Dict[str, Any] = {}

    @property
    def positions(self) -> Mapping[str, float]:
        """Read-only view of active position quantities {ticker: shares}."""
        return dict(self._positions)

    def set_target_weights(self, weights: Dict[str, float]) -> None:
        """
        Set target portfolio allocations w_i in [-1.0, 1.0], sum(|w_i|) <= 1.0.
        Executed at Open price of bar t+1.
        """
        if not isinstance(weights, dict):
            raise TypeError("Target weights must be a dictionary mapping ticker -> float.")
        self._target_weights = {str(k): float(v) for k, v in weights.items()}

    def set_target_weight(self, weight: float) -> None:
        """
        Convenience method for single-asset mode (Track B).
        Sets target allocation for context.symbol.
        """
        if self.symbol is None:
            raise ValueError(
                "context.set_target_weight(weight) is only available when context.symbol is defined. "
                "Use context.set_target_weights({ticker: weight}) for multi-asset universes."
            )
        self.set_target_weights({self.symbol: float(weight)})

    def get_target_weights(self) -> Dict[str, float]:
        """Returns the queued target weights."""
        return self._target_weights


class BaseStrategy(ABC):
    """
    Abstract Base Strategy Interface.
    All participant strategies must inherit from BaseStrategy and implement
    initialize() and on_bar().
    """

    @abstractmethod
    def initialize(self, context: Context) -> None:
        """
        Invoked once before simulation starts.
        Use this to declare indicators, parameters, lookback windows, and ML models.
        """
        pass

    @abstractmethod
    def on_bar(self, context: Context, bars: Dict[str, Bar]) -> None:
        """
        Invoked at each simulation step t.
        
        Parameters:
        - context: System state (portfolio_value, cash, positions, target weight setter).
        - bars: Dictionary mapping ticker string to immutable Bar object at current time t.
          In Track A, bars contains all 24 large-cap assets for that date.
          In Track B, bars contains a single entry with the active microstructure asset.
        """
        pass
