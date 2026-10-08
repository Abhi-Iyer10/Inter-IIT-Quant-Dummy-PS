"""
src/engine.py - Mandatory Participant Strategy Entry Point
Inter IIT Tech Meet 15.0 Quant Evaluation

Participants must implement their trading logic inside the ParticipantStrategy class.
This file provides a baseline random/signal sample strategy to test the complete pipeline.
"""

from __future__ import annotations
import math
import random
import numpy as np
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
        self.lookback_window: int = 252    # 12-month lookback window
        self.top_k: int = 5                # Top K asset selection
        self.skip_mom: int = 21            # Skip recent 1-month return (21 days)
        self.lookback_vol: int = 30        # 30-day realized volatility lookback
        self.w_mom: float = 0.7            # Weight for Momentum Z-score
        self.w_rev: float = 0.3            # Weight for Reversal Z-score

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
        if context.step % self.rebalance_cadence != 0 or context.step < self.lookback_window:
            return

        # 3. Sample Signal & Volatility Modeling
        # Computes 12m-1m Momentum & 1m Reversal signals, ranks assets,
        # selects top K, and scales weights inversely by volatility (vol-adjusted)
        mom_scores: Dict[str, float] = {}
        rev_scores: Dict[str, float] = {}

        for ticker, prices in self.history.items():
            if len(prices) >= self.lookback_window:
                p = np.array(prices, dtype=np.float64)
                
                # Signal 1: 12m - 1m Cross-Sectional Momentum
                ret_12m_1m = (p[-self.skip_mom] - p[-self.lookback_window]) / p[-self.lookback_window]
                mom_scores[ticker] = float(ret_12m_1m)

                # Signal 2: 1m Short-Term Reversal (-1 * 21-day return)
                ret_1m = (p[-1] - p[-self.skip_mom]) / p[-self.skip_mom]
                rev_scores[ticker] = float(-1.0 * ret_1m)

        if not mom_scores:
            return

        # Cross-sectional standardization (Z-score normalization)
        tickers = list(mom_scores.keys())
        mom_vals = np.array([mom_scores[t] for t in tickers], dtype=np.float64)
        rev_vals = np.array([rev_scores[t] for t in tickers], dtype=np.float64)

        mom_std = np.std(mom_vals)
        rev_std = np.std(rev_vals)

        mom_z = (mom_vals - np.mean(mom_vals)) / (mom_std + 1e-8) if mom_std > 0 else np.zeros_like(mom_vals)
        rev_z = (rev_vals - np.mean(rev_vals)) / (rev_std + 1e-8) if rev_std > 0 else np.zeros_like(rev_vals)

        composite_scores = {
            t: float(self.w_mom * mz + self.w_rev * rz)
            for t, mz, rz in zip(tickers, mom_z, rev_z)
        }

        # Select Top K assets
        top_assets = sorted(composite_scores.keys(), key=lambda x: composite_scores[x], reverse=True)[:self.top_k]

        # Inverse-volatility risk parity sizing
        raw_weights: Dict[str, float] = {}
        for ticker in top_assets:
            p_hist = self.history[ticker]
            if len(p_hist) >= self.lookback_vol:
                log_rets = np.diff(np.log(p_hist[-self.lookback_vol:]))
                vol = float(np.std(log_rets) * np.sqrt(252))
                vol = max(1e-4, vol)
            else:
                vol = 0.25
            raw_weights[ticker] = 1.0 / vol

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