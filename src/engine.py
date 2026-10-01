"""
src/engine.py - Mandatory Participant Strategy Entry Point
Inter IIT Tech Meet 15.0 Quant Evaluation

Participants must implement their trading logic inside the ParticipantStrategy class.
This file provides a baseline random/signal sample strategy to test the complete pipeline.
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Optional

from strategy_base import BaseStrategy, Context, Bar


class ParticipantStrategy(BaseStrategy):
    """
    Sample Participant Strategy implementing BaseStrategy.
    Automatically detects whether it is running on Track A (multi-asset)
    or Track B (single-asset high-frequency) and demonstrates compliant execution.
    """

    def initialize(self, context: Context) -> None:
        """
        Executed once before backtest simulation starts.
        Initialize your indicator histories, ML models, and parameters here.
        """
        random.seed(42)
        
        # Configuration & state tracking
        self.rebalance_cadence: int = 5    # Track A: rebalance portfolio every 5 days
        self.track_b_decision_interval: int = 10  # Track B: evaluate every 10 seconds
        
        # Memory storage for asset price history (sliding windows)
        self.history: Dict[str, List[float]] = {}
        self.lookback_window: int = 20

        print(f"[ParticipantStrategy] Initialized with universe: {len(context.universe)} assets.")
        if context.symbol:
            print(f"[ParticipantStrategy] Single-asset mode active for ticker: {context.symbol}")

    def on_bar(self, context: Context, bars: Dict[str, Bar]) -> None:
        """
        Executed at every simulation step t.
        
        Parameters:
        - context: Evaluation state (cash, equity, active holdings, target weight setters).
        - bars: Dictionary mapping ticker string to immutable Bar at time step t.
        """
        num_assets = len(bars)

        if num_assets > 1:
            # ==============================================================
            # Track A: Multi-Asset Cross-Sectional Strategy
            # ==============================================================
            self._handle_track_a(context, bars)
        else:
            # ==============================================================
            # Track B: Single-Asset High-Frequency Microstructure Strategy
            # ==============================================================
            self._handle_track_b(context, bars)

    def _handle_track_a(self, context: Context, bars: Dict[str, Bar]) -> None:
        """
        Track A Logic:
        - Maintains price histories
        - Rebalances every `rebalance_cadence` trading days
        - Generates volatility-adjusted target weights with gross leverage <= 1.0
        """
        # 1. Update historical close prices
        for ticker, bar in bars.items():
            if ticker not in self.history:
                self.history[ticker] = []
            self.history[ticker].append(bar.close)
            if len(self.history[ticker]) > self.lookback_window:
                self.history[ticker].pop(0)

        # 2. Only rebalance on defined cadence
        if context.step % self.rebalance_cadence != 0:
            return

        # 3. Sample Signal & Volatility Modeling
        # For demonstration: generates random buy/sell directional signals
        # and scales them inversely by sample standard deviation (vol-adjusted)
        raw_weights: Dict[str, float] = {}
        for ticker, bar in bars.items():
            prices = self.history.get(ticker, [])
            if len(prices) >= 2:
                # Simple return volatility proxy
                returns = [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))]
                mean_r = sum(returns) / len(returns)
                var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                vol = math.sqrt(var_r) if var_r > 1e-8 else 0.01
            else:
                vol = 0.01

            # Generate random score (-1.0 to +1.0)
            score = random.uniform(-1.0, 1.0)
            
            # Volatility-adjusted inverse sizing: w ~ score / vol
            raw_weights[ticker] = score / vol

        # 4. Normalize weights so gross leverage sum(|w_i|) <= 1.0 (strict rule)
        total_abs_weight = sum(abs(w) for w in raw_weights.values())
        if total_abs_weight > 0.0:
            target_weights = {ticker: w / total_abs_weight for ticker, w in raw_weights.items()}
        else:
            target_weights = {ticker: 0.0 for ticker in raw_weights}

        # 5. Submit target allocations to context
        context.set_target_weights(target_weights)

    def _handle_track_b(self, context: Context, bars: Dict[str, Bar]) -> None:
        """
        Track B Logic:
        - Continuous high-frequency microstructure stream
        - Accesses microstructure quant fields (OFI, VWAP, Log-Range)
        - Adapts position sizing to order flow imbalances and volatility
        """
        ticker = context.symbol or next(iter(bars.keys()))
        bar = bars[ticker]
        t = context.step

        # 1. Check capital constraint: stop trading if account equity is depleted
        if context.portfolio_value <= 0.0:
            context.set_target_weights({ticker: 0.0})
            return

        # 2. Only re-evaluate at decision intervals to simulate discrete trade decisions
        if t % self.track_b_decision_interval != 0:
            return

        # 3. Read Microstructure quant indicators
        ofi = bar.ofi                # Order Flow Imbalance in [-1.0, 1.0]
        vwap = bar.vwap              # Volume-Weighted Average Price
        log_range = bar.log_range    # Volatility proxy: ln(H) - ln(L)

        # Baseline sample decision:
        # Example: if OFI indicates buyer pressure (>0.2) -> buy, if seller pressure (<-0.2) -> sell
        # Or sample position within gross leverage limit [-0.5, 0.5]
        if ofi > 0.2:
            target_weight = 0.5   # 50% long
        elif ofi < -0.2:
            target_weight = -0.5  # 50% short
        else:
            target_weight = random.choice([-0.3, 0.0, 0.3])

        context.set_target_weights({ticker: target_weight})
