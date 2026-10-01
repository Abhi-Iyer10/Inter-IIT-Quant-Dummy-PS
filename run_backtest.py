"""
run_backtest.py - Command Line Execution Runner for Participants & Evaluators
Inter IIT Tech Meet 15.0 Quant Evaluation

Usage:
  python run_backtest.py --track A
  python run_backtest.py --track B --max-ticks 86400
  python run_backtest.py --track B
"""

from __future__ import annotations
import os
import sys
import argparse
import importlib
import yaml

from backtester import Backtester, BacktestConfig
from strategy_base import BaseStrategy


def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def format_table(metrics: dict) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append(f"{'QUANT BACKTEST EVALUATION SUMMARY':^65}")
    lines.append("=" * 65)

    sections = [
        ("PORTFOLIO PERFORMANCE", [
            "Track", "Total Steps", "Execution Duration (s)", "Throughput (ticks/s)",
            "Initial Cash", "Final Equity", "Total Return (%)", "CAGR (%)",
            "Sharpe Ratio", "Sharpe 95% CI", "Sortino Ratio", "Max Drawdown (%)",
            "Calmar Ratio", "Daily Win Rate (%)"
        ]),
        ("TRADING ACTIVITY & CADENCE", [
            "Total Trades", "Buy Trades", "Sell Trades",
            "Trading Frequency (trades/day)", "Average Trade Gap (steps)",
            "Median Trade Gap (steps)", "Min Trade Gap (steps)", "Max Trade Gap (steps)",
            "Market Exposure Time (%)"
        ]),
        ("EXPOSURE & RISK CONSTRAINTS", [
            "Mean Gross Leverage", "Max Gross Leverage",
            "Gross Leverage Violations", "Account Bankrupt (Capital <= 0)"
        ]),
        ("TURNOVER & TRANSACTION COSTS", [
            "Total Turnover (%)", "Total Turnover Value ($)",
            "Average Trade Value ($)", "Total Commission Paid ($)", "Total Slippage Paid ($)"
        ]),
    ]

    used_keys = set()
    for section_title, keys in sections:
        section_lines = []
        for k in keys:
            if k in metrics:
                used_keys.add(k)
                v = metrics[k]
                if isinstance(v, float):
                    val_str = f"{v:,.2f}"
                elif isinstance(v, list):
                    val_str = f"[{v[0]:.2f}, {v[1]:.2f}]"
                elif isinstance(v, int):
                    val_str = f"{v:,}"
                else:
                    val_str = str(v)
                section_lines.append(f"  {k:<35}: {val_str:>24}")
        if section_lines:
            lines.append(f"-- {section_title} " + "-" * max(2, 62 - len(section_title)))
            lines.extend(section_lines)

    # Any remaining keys not in categorized sections
    remaining = [k for k in metrics if k not in used_keys]
    if remaining:
        lines.append("-- OTHER METRICS " + "-" * 48)
        for k in remaining:
            v = metrics[k]
            if isinstance(v, float):
                val_str = f"{v:,.2f}"
            elif isinstance(v, list):
                val_str = f"[{v[0]:.2f}, {v[1]:.2f}]"
            elif isinstance(v, int):
                val_str = f"{v:,}"
            else:
                val_str = str(v)
            lines.append(f"  {k:<35}: {val_str:>24}")

    lines.append("=" * 65)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run Quant Backtester on Strategy.")
    parser.add_argument("--track", choices=["A", "B", "a", "b"], default=None,
                        help="Track to evaluate ('A' for cross-sectional daily, 'B' for 1s microstructure).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--data", default=None, help="Path to input dataset file.")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="Optional limit on ticks (Track B) for rapid local development.")
    parser.add_argument("--initial-cash", type=float, default=None, help="Starting cash balance.")
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib tearsheet generation.")
    parser.add_argument("--output-dir", default=None, help="Directory to save summary and tearsheet.")
    args = parser.parse_args()

    cfg_file = load_config(args.config)

    # Determine track
    track = (args.track or cfg_file.get("track", "A")).upper()

    initial_cash = args.initial_cash or float(cfg_file.get("initial_cash", 100_000.0))
    slippage = float(cfg_file.get("slippage_rate", 0.0005))
    commission = float(cfg_file.get("commission_rate", 0.0001))
    output_dir = args.output_dir or cfg_file.get("output_dir", "results")
    generate_plot = not args.no_plot and bool(cfg_file.get("generate_plot", True))

    if track == "A":
        t_cfg = cfg_file.get("track_a", {})
        data_path = args.data or t_cfg.get("data_path", "data/Dataset_PS-A.csv")
        symbol = "ALL"
        max_ticks = None
    else:
        t_cfg = cfg_file.get("track_b", {})
        data_path = args.data or t_cfg.get("data_path", "data/ASSET_ALPHA_1s.parquet")
        symbol = t_cfg.get("symbol", "ALPHA")
        max_ticks = args.max_ticks if args.max_ticks is not None else t_cfg.get("max_ticks")

    config = BacktestConfig(
        track=track,
        data_path=data_path,
        initial_cash=initial_cash,
        slippage_rate=slippage,
        commission_rate=commission,
        symbol=symbol,
        max_ticks=max_ticks,
        save_results=True,
        output_dir=output_dir,
        generate_plot=generate_plot
    )

    # Load participant strategy from src.engine
    sys.path.insert(0, os.path.abspath("."))
    try:
        engine_mod = importlib.import_module("src.engine")
        strategy_cls = getattr(engine_mod, "ParticipantStrategy", None)
        if strategy_cls is None:
            raise AttributeError("src/engine.py must define a class named 'ParticipantStrategy'.")
        if not issubclass(strategy_cls, BaseStrategy):
            raise TypeError("ParticipantStrategy must inherit from strategy_base.BaseStrategy.")
    except Exception as e:
        print(f"[Error] Failed to load strategy from src/engine.py: {e}")
        sys.exit(1)

    strategy = strategy_cls()

    # Run Backtest
    backtester = Backtester(config)
    result = backtester.run(strategy)

    # Print summary
    print("\n" + format_table(result.metrics_summary) + "\n")


if __name__ == "__main__":
    main()
