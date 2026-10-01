"""
backtester.py - Canonical High-Throughput Event-Driven Backtesting Core
Inter IIT Tech Meet 15.0 Quant Evaluation Architecture

Supports:
- Track A: Cross-Sectional Multi-Asset Daily Rebalancing
- Track B: High-Frequency (1s) Microstructure Regime Switching (> 200k ticks/sec)
- Strict Non-Negative Capital & No-Leverage Enforcement
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from strategy_base import Bar, FastBarView, Context, BaseStrategy


@dataclass
class BacktestConfig:
    track: str = "A"                     # "A" or "B"
    data_path: str = ""
    initial_cash: float = 100_000.0
    slippage_rate: float = 0.0005        # 5 bps execution slippage
    commission_rate: float = 0.0001      # 1 bp transaction commission
    symbol: str = "ALPHA"                # Single asset symbol for Track B
    max_ticks: Optional[int] = None      # Optional tick limit (for fast testing)
    save_results: bool = True
    output_dir: str = "results"
    generate_plot: bool = True


@dataclass
class BacktestResult:
    track: str
    total_steps: int
    duration_seconds: float
    throughput_ticks_per_sec: float
    initial_cash: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sharpe_ci_95: Tuple[float, float]
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    leverage_violations: int
    is_bankrupt: bool
    metrics_summary: Dict[str, Any] = field(default_factory=dict)


class RuleGuard:
    """
    Constraint & Rule Enforcer:
    - Gross leverage <= 1.0 (no borrowing)
    - Strict Non-Negative Capital Constraint (capital cannot be negative; trading halts if depleted)
    """

    def __init__(self, track: str):
        self.track = track.upper()
        self.leverage_violations: int = 0

    def enforce(
        self,
        step: int,
        timestamp: Any,
        target_weights: Dict[str, float],
        current_positions: Mapping[str, float],
        current_equity: float,
        is_bankrupt: bool = False
    ) -> Dict[str, float]:
        # 1. Capital Depletion / Bankruptcy Check:
        # If account equity <= 0, no trades can be taken.
        if is_bankrupt or current_equity <= 0.0:
            return {k: 0.0 for k in target_weights}

        adjusted_weights = dict(target_weights)

        # Sanitize non-finite values (NaN / Inf)
        for k, v in list(adjusted_weights.items()):
            if not math.isfinite(v):
                adjusted_weights[k] = 0.0

        # 2. Gross Leverage Check: sum(|w_i|) <= 1.0
        gross_leverage = sum(abs(w) for w in adjusted_weights.values())
        if gross_leverage > 1.000001:
            self.leverage_violations += 1
            # Scale down proportionally to 1.0 (strictly no borrowing/leverage)
            scale = 1.0 / gross_leverage
            adjusted_weights = {k: v * scale for k, v in adjusted_weights.items()}

        return adjusted_weights


class AccountingLedger:
    """
    Portfolio & Accounting Ledger:
    - Tracks cash, holdings, equity, and leverage
    - Enforces that capital can NEVER be negative (capital >= 0.0 at all times)
    - Executes orders at t+1 Open with slippage and commission
    - Marks-to-market at t+1 Close
    """

    def __init__(
        self,
        initial_cash: float,
        slippage_rate: float,
        commission_rate: float
    ):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.equity = float(initial_cash)
        self.slippage_rate = float(slippage_rate)
        self.commission_rate = float(commission_rate)
        self.positions: Dict[str, float] = {}
        self.is_bankrupt: bool = False

    def execute_pending_orders(
        self,
        target_weights: Dict[str, float],
        open_prices: Dict[str, float],
        base_equity: float
    ) -> None:
        """
        Fill orders at t+1 Open price based on target weights decided at t.
        DeltaShares_i = (Equity_t * w_i / Open_{i, t+1}) - Shares_{i, t}
        Strictly bounds purchases by available cash so cash can never be negative.
        """
        if self.is_bankrupt or base_equity <= 0.0:
            self.positions.clear()
            return

        all_tickers = set(self.positions.keys()) | set(target_weights.keys())

        # First handle sell/reduction orders to free up cash, then buy orders
        sell_orders = []
        buy_orders = []

        for ticker in all_tickers:
            open_price = open_prices.get(ticker)
            if open_price is None or open_price <= 0.0:
                continue

            target_weight = target_weights.get(ticker, 0.0)
            current_shares = self.positions.get(ticker, 0.0)

            desired_shares = (base_equity * target_weight) / open_price
            delta_shares = desired_shares - current_shares

            if abs(delta_shares) < 1e-8:
                continue

            if delta_shares < 0:
                sell_orders.append((ticker, delta_shares, open_price))
            else:
                buy_orders.append((ticker, delta_shares, open_price))

        # Execute sell orders first (generating cash)
        for ticker, delta_shares, open_price in sell_orders:
            fill_price = open_price * (1.0 - self.slippage_rate)
            proceeds = abs(delta_shares) * fill_price
            commission = proceeds * self.commission_rate
            net_cash_change = proceeds - commission
            
            self.cash = max(0.0, self.cash + net_cash_change)
            
            new_shares = self.positions.get(ticker, 0.0) + delta_shares
            if abs(new_shares) < 1e-8:
                self.positions.pop(ticker, None)
            else:
                self.positions[ticker] = new_shares

        # Execute buy orders (bounded strictly by available cash - no negative cash/borrowing)
        for ticker, delta_shares, open_price in buy_orders:
            fill_price = open_price * (1.0 + self.slippage_rate)
            cost_per_share = fill_price * (1.0 + self.commission_rate)
            
            if cost_per_share <= 0.0:
                continue

            # Ensure purchase does not exceed available unborrowed cash
            max_affordable_shares = max(0.0, self.cash / cost_per_share)
            executed_shares = min(delta_shares, max_affordable_shares)

            if executed_shares < 1e-8:
                continue

            trade_value = executed_shares * fill_price
            commission = trade_value * self.commission_rate
            total_outflow = trade_value + commission

            self.cash = max(0.0, self.cash - total_outflow)

            new_shares = self.positions.get(ticker, 0.0) + executed_shares
            if abs(new_shares) < 1e-8:
                self.positions.pop(ticker, None)
            else:
                self.positions[ticker] = new_shares

    def mark_to_market(self, close_prices: Dict[str, float]) -> float:
        """
        Compute total marked-to-market equity at Close:
        Equity = Cash + sum(Shares_i * Close_i)
        Enforces non-negative capital constraint: if equity <= 0, account is bankrupt,
        capital is floored at 0.0, and positions are liquidated.
        """
        if self.is_bankrupt:
            self.equity = 0.0
            self.cash = 0.0
            self.positions.clear()
            return 0.0

        holdings_value = 0.0
        for ticker, shares in self.positions.items():
            close_price = close_prices.get(ticker, 0.0)
            holdings_value += shares * close_price

        raw_equity = self.cash + holdings_value

        # Capital cannot be negative constraint
        if raw_equity <= 0.0:
            self.equity = 0.0
            self.cash = 0.0
            self.positions.clear()
            self.is_bankrupt = True
        else:
            self.equity = raw_equity

        return self.equity


class MetricsEngine:
    """
    Computes quantitative performance metrics & statistical validation:
    - Annualized Sharpe Ratio
    - Stationary Block-Bootstrap 95% Confidence Interval for Sharpe
    - Downside Sortino Ratio
    - Maximum Drawdown & Calmar Ratio
    - Leverage violations and bankruptcy compliance
    """

    @staticmethod
    def compute(
        equity_curve: np.ndarray,
        track: str,
        initial_cash: float,
        start_time: float,
        end_time: float,
        leverage_violations: int,
        is_bankrupt: bool,
        steps_per_day: int = 1
    ) -> BacktestResult:
        total_steps = len(equity_curve)
        duration = max(1e-4, end_time - start_time)
        throughput = total_steps / duration
        final_equity = float(equity_curve[-1])
        total_return_pct = ((final_equity / initial_cash) - 1.0) * 100.0

        if total_steps < 2:
            return BacktestResult(
                track=track, total_steps=total_steps, duration_seconds=duration,
                throughput_ticks_per_sec=throughput, initial_cash=initial_cash,
                final_equity=final_equity, total_return_pct=0.0, cagr_pct=0.0,
                sharpe_ratio=0.0, sharpe_ci_95=(0.0, 0.0), sortino_ratio=0.0,
                max_drawdown_pct=0.0, calmar_ratio=0.0,
                leverage_violations=leverage_violations, is_bankrupt=is_bankrupt
            )

        # For Track B (1s frequency, steps_per_day=86400), resample returns to daily bars
        # to calculate statistically sound Sharpe and avoid micro-autocorrelation bias
        if track.upper() == "B" and steps_per_day > 1:
            daily_equities = equity_curve[::steps_per_day]
            if len(daily_equities) < 2:
                daily_equities = np.array([equity_curve[0], equity_curve[-1]])
            returns = np.diff(daily_equities) / np.maximum(1e-8, daily_equities[:-1])
            annual_factor = 252.0
            days_elapsed = total_steps / steps_per_day
        else:
            returns = np.diff(equity_curve) / np.maximum(1e-8, equity_curve[:-1])
            annual_factor = 252.0
            days_elapsed = max(1.0, total_steps)

        # Clean NaNs or Infs
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            returns = np.zeros(1)

        # CAGR
        years = max(1e-3, days_elapsed / 252.0)
        cagr_pct = ((final_equity / initial_cash) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

        # Sharpe Ratio
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        if std_ret > 1e-12:
            sharpe_ratio = float((mean_ret / std_ret) * math.sqrt(annual_factor))
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio
        downside_returns = returns[returns < 0.0]
        downside_std = float(np.std(downside_returns, ddof=1)) if len(downside_returns) > 1 else 0.0
        if downside_std > 1e-12:
            sortino_ratio = float((mean_ret / downside_std) * math.sqrt(annual_factor))
        else:
            sortino_ratio = 0.0

        # Max Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / np.maximum(1e-8, running_max)
        max_drawdown_pct = float(abs(np.min(drawdowns))) * 100.0

        # Calmar Ratio
        calmar_ratio = float(cagr_pct / max_drawdown_pct) if max_drawdown_pct > 1e-4 else 0.0

        # Stationary Block-Bootstrap 95% Confidence Interval for Sharpe
        sharpe_ci = MetricsEngine._bootstrap_sharpe_ci(returns, annual_factor, num_samples=1000)

        metrics_summary = {
            "Track": track,
            "Total Steps": total_steps,
            "Execution Duration (s)": round(duration, 3),
            "Throughput (ticks/s)": int(throughput),
            "Initial Cash": initial_cash,
            "Final Equity": round(final_equity, 2),
            "Total Return (%)": round(total_return_pct, 2),
            "CAGR (%)": round(cagr_pct, 2),
            "Sharpe Ratio": round(sharpe_ratio, 3),
            "Sharpe 95% CI": [round(sharpe_ci[0], 3), round(sharpe_ci[1], 3)],
            "Sortino Ratio": round(sortino_ratio, 3),
            "Max Drawdown (%)": round(max_drawdown_pct, 2),
            "Calmar Ratio": round(calmar_ratio, 3),
            "Gross Leverage Violations": leverage_violations,
            "Account Bankrupt (Capital <= 0)": is_bankrupt,
        }

        return BacktestResult(
            track=track,
            total_steps=total_steps,
            duration_seconds=duration,
            throughput_ticks_per_sec=throughput,
            initial_cash=initial_cash,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            cagr_pct=cagr_pct,
            sharpe_ratio=sharpe_ratio,
            sharpe_ci_95=sharpe_ci,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            calmar_ratio=calmar_ratio,
            leverage_violations=leverage_violations,
            is_bankrupt=is_bankrupt,
            metrics_summary=metrics_summary,
        )

    @staticmethod
    def _bootstrap_sharpe_ci(
        returns: np.ndarray,
        annual_factor: float,
        num_samples: int = 1000,
        block_size: int = 5
    ) -> Tuple[float, float]:
        """
        Block bootstrap to preserve time-series dependency when computing Sharpe CI.
        """
        n = len(returns)
        if n < 10:
            return (0.0, 0.0)

        num_blocks = max(1, n // block_size)
        boot_sharpes = []
        rng = np.random.default_rng(42)

        for _ in range(num_samples):
            block_starts = rng.integers(0, max(1, n - block_size + 1), size=num_blocks)
            sampled_idx = np.concatenate([np.arange(idx, idx + block_size) for idx in block_starts])[:n]
            sample = returns[sampled_idx]
            
            s_mean = np.mean(sample)
            s_std = np.std(sample, ddof=1)
            if s_std > 1e-12:
                b_sharpe = (s_mean / s_std) * math.sqrt(annual_factor)
                boot_sharpes.append(b_sharpe)

        if not boot_sharpes:
            return (0.0, 0.0)

        lower_ci = float(np.percentile(boot_sharpes, 2.5))
        upper_ci = float(np.percentile(boot_sharpes, 97.5))
        return (lower_ci, upper_ci)


class Backtester:
    """
    Main Orchestrator Engine for Backtesting.
    Executes Track A (Multi-asset daily) and Track B (Continuous High-frequency 1s).
    Enforces non-negative capital and zero leverage.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.rule_guard = RuleGuard(track=config.track)
        self.ledger = AccountingLedger(
            initial_cash=config.initial_cash,
            slippage_rate=config.slippage_rate,
            commission_rate=config.commission_rate
        )

    def run(self, strategy: BaseStrategy) -> BacktestResult:
        if self.config.track.upper() == "A":
            return self._run_track_a(strategy)
        elif self.config.track.upper() == "B":
            return self._run_track_b(strategy)
        else:
            raise ValueError(f"Unknown track: {self.config.track}. Must be 'A' or 'B'.")

    def _run_track_a(self, strategy: BaseStrategy) -> BacktestResult:
        """
        Executes Track A: Universe of 24 assets, daily OHLCV, cross-sectional packages.
        """
        csv_path = self.config.data_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Track A dataset not found: {csv_path}")

        print(f"[Backtester] Loading Track A data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df['date'] = df['date'].astype(str)
        df = df.sort_values(by=['date', 'ticker']).reset_index(drop=True)

        dates = df['date'].unique().tolist()
        universe = sorted(df['ticker'].unique().tolist())
        print(f"[Backtester] Track A initialized: {len(dates)} trading days, {len(universe)} assets.")

        context = Context(initial_cash=self.config.initial_cash, universe=universe)
        strategy.initialize(context)

        # Pre-group bars by date into in-memory dictionaries for speed
        grouped = df.groupby('date')
        daily_bars_list: List[Tuple[str, Dict[str, Bar], Dict[str, float], Dict[str, float]]] = []

        for date_str, group in grouped:
            bars_dict: Dict[str, Bar] = {}
            open_dict: Dict[str, float] = {}
            close_dict: Dict[str, float] = {}
            
            for row in group.itertuples(index=False):
                ticker = row.ticker
                b = Bar(
                    ticker=ticker,
                    timestamp=date_str,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    volume=row.volume
                )
                bars_dict[ticker] = b
                open_dict[ticker] = float(row.open)
                close_dict[ticker] = float(row.close)

            daily_bars_list.append((date_str, bars_dict, open_dict, close_dict))

        total_steps = len(daily_bars_list)
        equity_curve = np.zeros(total_steps, dtype=np.float64)
        leverage_curve = np.zeros(total_steps, dtype=np.float64)

        pending_target_weights: Dict[str, float] = {}
        pending_base_equity: float = self.config.initial_cash

        start_time = time.perf_counter()

        for step_idx, (date_str, bars_dict, open_prices, close_prices) in enumerate(daily_bars_list):
            # Step t+1 Open: Execute pending target orders generated at step t
            if pending_target_weights and not self.ledger.is_bankrupt:
                self.ledger.execute_pending_orders(
                    target_weights=pending_target_weights,
                    open_prices=open_prices,
                    base_equity=pending_base_equity
                )

            # Step t Mark-to-Market at Close
            current_equity = self.ledger.mark_to_market(close_prices)
            equity_curve[step_idx] = current_equity

            # Current leverage
            total_holdings_val = sum(abs(q * close_prices.get(k, 0.0)) for k, q in self.ledger.positions.items())
            leverage_curve[step_idx] = total_holdings_val / max(1.0, current_equity)

            # Sync context for strategy invocation
            context.step = step_idx
            context.timestamp = date_str
            context.cash = self.ledger.cash
            context.portfolio_value = current_equity
            context._positions = dict(self.ledger.positions)

            # Strategy on_bar invocation (receives full package of 24 assets)
            if not self.ledger.is_bankrupt:
                strategy.on_bar(context, bars_dict)
                raw_weights = context.get_target_weights()
            else:
                raw_weights = {}

            # Extract and validate target weights via RuleGuard
            guarded_weights = self.rule_guard.enforce(
                step=step_idx,
                timestamp=date_str,
                target_weights=raw_weights,
                current_positions=self.ledger.positions,
                current_equity=current_equity,
                is_bankrupt=self.ledger.is_bankrupt
            )

            # Queue for execution at step t+1 Open
            pending_target_weights = guarded_weights
            pending_base_equity = current_equity

        end_time = time.perf_counter()

        result = MetricsEngine.compute(
            equity_curve=equity_curve,
            track="A",
            initial_cash=self.config.initial_cash,
            start_time=start_time,
            end_time=end_time,
            leverage_violations=self.rule_guard.leverage_violations,
            is_bankrupt=self.ledger.is_bankrupt,
            steps_per_day=1
        )

        if self.config.save_results:
            self._save_output(result, equity_curve, leverage_curve, dates)

        return result

    def _run_track_b(self, strategy: BaseStrategy) -> BacktestResult:
        """
        Executes Track B: Ultra-high throughput streaming of 1-second ticks.
        Continuous 24-hour microstructure execution without artificial EOD stops.
        Employs contiguous NumPy arrays and zero-allocation FastBarView.
        """
        parquet_path = self.config.data_path
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Track B dataset not found: {parquet_path}")

        print(f"[Backtester] Loading Track B parquet from {parquet_path}...")
        table = pq.read_table(parquet_path)
        num_rows = table.num_rows

        if self.config.max_ticks is not None and self.config.max_ticks > 0:
            num_rows = min(num_rows, self.config.max_ticks)
            table = table.slice(0, num_rows)

        print(f"[Backtester] Unpacking {num_rows:,} ticks into contiguous NumPy buffers...")
        open_arr = table['open'].to_numpy().astype(np.float64)
        high_arr = table['high'].to_numpy().astype(np.float64)
        low_arr = table['low'].to_numpy().astype(np.float64)
        close_arr = table['close'].to_numpy().astype(np.float64)
        vol_arr = table['volume'].to_numpy().astype(np.float64)
        qvol_arr = table['quote_volume'].to_numpy().astype(np.float64)
        count_arr = table['count'].to_numpy().astype(np.int64)
        tbv_arr = table['taker_buy_volume'].to_numpy().astype(np.float64)

        symbol = self.config.symbol
        context = Context(initial_cash=self.config.initial_cash, universe=[symbol], symbol=symbol)
        strategy.initialize(context)

        # Pre-allocate flyweight bar and reusable single-entry dictionary
        flyweight_bar = FastBarView(
            ticker=symbol, timestamp=0, open=open_arr[0], high=high_arr[0],
            low=low_arr[0], close=close_arr[0], volume=vol_arr[0],
            quote_volume=qvol_arr[0], count=count_arr[0], taker_buy_volume=tbv_arr[0]
        )
        bars_dict: Dict[str, Bar] = {symbol: flyweight_bar}
        open_prices: Dict[str, float] = {symbol: open_arr[0]}
        close_prices: Dict[str, float] = {symbol: close_arr[0]}

        equity_curve = np.zeros(num_rows, dtype=np.float64)
        leverage_curve = np.zeros(num_rows, dtype=np.float64)

        pending_target_weights: Dict[str, float] = {}
        pending_base_equity: float = self.config.initial_cash

        print(f"[Backtester] Executing Track B streaming loop for {num_rows:,} ticks...")
        start_time = time.perf_counter()

        for t in range(num_rows):
            o_val = open_arr[t]
            h_val = high_arr[t]
            l_val = low_arr[t]
            c_val = close_arr[t]
            v_val = vol_arr[t]
            qv_val = qvol_arr[t]
            cnt_val = count_arr[t]
            tbv_val = tbv_arr[t]

            flyweight_bar.update(
                ticker=symbol, timestamp=t, open_val=o_val, high_val=h_val,
                low_val=l_val, close_val=c_val, vol_val=v_val,
                qvol_val=qv_val, count_val=cnt_val, tbv_val=tbv_val
            )
            open_prices[symbol] = o_val
            close_prices[symbol] = c_val

            # Step t Open: execute pending target allocation from t-1
            if pending_target_weights and not self.ledger.is_bankrupt:
                self.ledger.execute_pending_orders(
                    target_weights=pending_target_weights,
                    open_prices=open_prices,
                    base_equity=pending_base_equity
                )

            # Step t Mark-to-Market at Close
            current_equity = self.ledger.mark_to_market(close_prices)
            equity_curve[t] = current_equity

            # Current leverage
            current_shares = self.ledger.positions.get(symbol, 0.0)
            leverage_curve[t] = abs(current_shares * c_val) / max(1.0, current_equity)

            # Update context state
            context.step = t
            context.timestamp = t
            context.cash = self.ledger.cash
            context.portfolio_value = current_equity
            context._positions = dict(self.ledger.positions)

            # Invoke strategy
            if not self.ledger.is_bankrupt:
                strategy.on_bar(context, bars_dict)
                raw_weights = context.get_target_weights()
            else:
                raw_weights = {}

            # RuleGuard enforcement (Gross Leverage + Non-Negative Capital)
            guarded_weights = self.rule_guard.enforce(
                step=t,
                timestamp=t,
                target_weights=raw_weights,
                current_positions=self.ledger.positions,
                current_equity=current_equity,
                is_bankrupt=self.ledger.is_bankrupt
            )

            pending_target_weights = guarded_weights
            pending_base_equity = current_equity

        end_time = time.perf_counter()

        result = MetricsEngine.compute(
            equity_curve=equity_curve,
            track="B",
            initial_cash=self.config.initial_cash,
            start_time=start_time,
            end_time=end_time,
            leverage_violations=self.rule_guard.leverage_violations,
            is_bankrupt=self.ledger.is_bankrupt,
            steps_per_day=86400
        )

        if self.config.save_results:
            self._save_output(result, equity_curve, leverage_curve, timestamps=None)

        return result

    def _save_output(
        self,
        result: BacktestResult,
        equity_curve: np.ndarray,
        leverage_curve: np.ndarray,
        timestamps: Optional[List[Any]] = None
    ) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 1. Save summary.json
        summary_path = os.path.join(self.config.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(result.metrics_summary, f, indent=2)
        print(f"[Backtester] Saved summary metrics to {summary_path}")

        # 2. Save tearsheet.png if requested
        if self.config.generate_plot:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                steps = np.arange(len(equity_curve))

                # Plot 1: Equity Curve
                axes[0].plot(steps, equity_curve, color="#1f77b4", lw=1.5, label="Portfolio Equity ($)")
                axes[0].axhline(self.config.initial_cash, color="gray", linestyle="--", alpha=0.7)
                bankrupt_tag = " [BANKRUPT]" if result.is_bankrupt else ""
                axes[0].set_title(
                    f"Track {result.track} - Equity Curve | Final: ${result.final_equity:,.2f} "
                    f"({result.total_return_pct:+.2f}%){bankrupt_tag}"
                )
                axes[0].set_ylabel("Equity ($)")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc="upper left")

                # Plot 2: Drawdown
                running_max = np.maximum.accumulate(equity_curve)
                drawdowns = ((equity_curve - running_max) / np.maximum(1e-8, running_max)) * 100.0
                axes[1].fill_between(steps, drawdowns, 0, color="#d62728", alpha=0.4, label="Drawdown (%)")
                axes[1].plot(steps, drawdowns, color="#d62728", lw=1.0)
                axes[1].set_title(f"Underwater Drawdown | Max: {result.max_drawdown_pct:.2f}%")
                axes[1].set_ylabel("Drawdown (%)")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc="lower left")

                # Plot 3: Gross Leverage
                axes[2].plot(steps, leverage_curve, color="#2ca02c", lw=1.2, label="Gross Leverage")
                axes[2].axhline(1.0, color="red", linestyle="--", label="Leverage Limit (1.0)")
                axes[2].set_title(f"Gross Leverage Exposure | Violations: {result.leverage_violations}")
                axes[2].set_ylabel("Leverage (x)")
                axes[2].set_xlabel("Simulation Step")
                axes[2].set_ylim(-0.05, 1.25)
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(loc="upper left")

                plt.tight_layout()
                plot_path = os.path.join(self.config.output_dir, "tearsheet.png")
                plt.savefig(plot_path, dpi=150)
                plt.close(fig)
                print(f"[Backtester] Generated performance tearsheet: {plot_path}")
            except Exception as e:
                print(f"[Backtester] Warning: Failed to generate tearsheet plot: {e}")
