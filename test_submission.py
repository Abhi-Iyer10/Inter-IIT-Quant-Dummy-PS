"""
test_submission.py - Automated Pre-Submission Sanity Validator
Inter IIT Tech Meet 15.0 Quant Evaluation

Runs sanity checks on the participant repository and strategy:
1. File structure and importability
2. API contract adherence (initialize and on_bar)
3. Track A multi-asset cross-sectional bar handling
4. Track B continuous high-frequency microstructure handling
5. Non-negative capital and zero-leverage constraint verification
"""

import sys
import os
import math

from strategy_base import BaseStrategy, Context, Bar
from backtester import RuleGuard, AccountingLedger


def test_file_structure():
    print("[1/5] Checking file structure...")
    required_files = [
        "src/engine.py",
        "strategy_base.py",
        "backtester.py",
        "run_backtest.py",
        "requirements.txt",
        "config.yaml",
    ]
    for rf in required_files:
        if not os.path.exists(rf):
            raise FileNotFoundError(f"Missing required file: {rf}")
    print("  -> Passed.")


def test_strategy_class():
    print("[2/5] Checking strategy class definition and inheritance...")
    sys.path.insert(0, os.path.abspath("."))
    from src.engine import ParticipantStrategy

    if not issubclass(ParticipantStrategy, BaseStrategy):
        raise TypeError("ParticipantStrategy must inherit from strategy_base.BaseStrategy.")

    strat = ParticipantStrategy()
    if not hasattr(strat, "initialize") or not hasattr(strat, "on_bar"):
        raise AttributeError("ParticipantStrategy must implement initialize() and on_bar().")
    print("  -> Passed.")


def test_track_a_dummy():
    print("[3/5] Testing Track A cross-sectional bar handling...")
    from src.engine import ParticipantStrategy

    strat = ParticipantStrategy()
    tickers = [f"ASSET_{i:02d}" for i in range(24)]
    context = Context(initial_cash=100_000.0, universe=tickers)
    strat.initialize(context)

    # Create dummy bars for 10 dates
    for day in range(10):
        context.step = day
        context.timestamp = f"2026-01-{day+1:02d}"
        dummy_bars = {
            t: Bar(
                ticker=t,
                timestamp=context.timestamp,
                open=100.0 + day,
                high=105.0 + day,
                low=95.0 + day,
                close=102.0 + day,
                volume=1_000_000.0
            )
            for t in tickers
        }
        strat.on_bar(context, dummy_bars)
        weights = context.get_target_weights()
        if weights:
            gross_lev = sum(abs(float(w)) for w in weights.values())
            if gross_lev > 1.0001:
                raise ValueError(f"Leverage limit exceeded in Track A! Got gross leverage {gross_lev:.4f} > 1.0.")
    print("  -> Passed.")


def test_track_b_dummy():
    print("[4/5] Testing Track B continuous microstructure handling...")
    from src.engine import ParticipantStrategy

    strat = ParticipantStrategy()
    symbol = "ALPHA"
    context = Context(initial_cash=100_000.0, universe=[symbol], symbol=symbol)
    strat.initialize(context)

    # Test continuous streaming ticks
    for t in [10, 20, 100, 86400, 100000]:
        dummy_bar = Bar(
            ticker=symbol,
            timestamp=t,
            open=70000.0,
            high=70010.0,
            low=69990.0,
            close=70005.0,
            volume=10.0,
            quote_volume=700050.0,
            count=50,
            taker_buy_volume=6.0
        )
        context.step = t
        strat.on_bar(context, {symbol: dummy_bar})
        weights = context.get_target_weights()
        if weights:
            lev = sum(abs(float(w)) for w in weights.values())
            if lev > 1.0001:
                raise ValueError(f"Leverage limit exceeded in Track B! Got {lev:.4f} > 1.0.")

        # Test microstructure properties
        assert -1.0 <= dummy_bar.ofi <= 1.0, "OFI out of bounds"
        assert dummy_bar.vwap > 0.0, "VWAP must be positive"
        assert dummy_bar.log_range >= 0.0, "Log range must be non-negative"
    print("  -> Passed.")


def test_non_negative_capital_and_leverage():
    print("[5/5] Testing non-negative capital constraint and zero-leverage enforcement...")
    
    # 1. Test RuleGuard leverage clamping
    guard = RuleGuard(track="A")
    raw_weights = {"A": 0.8, "B": 0.6}  # Sum = 1.4 > 1.0
    clamped = guard.enforce(
        step=0, timestamp="2026-01-01", target_weights=raw_weights,
        current_positions={}, current_equity=100_000.0, is_bankrupt=False
    )
    clamped_lev = sum(abs(w) for w in clamped.values())
    assert clamped_lev <= 1.000001, f"Clamped leverage {clamped_lev} must not exceed 1.0"
    assert guard.leverage_violations == 1, "Violation counter must increment"

    # 2. Test RuleGuard capital depletion / bankruptcy clamping
    bankrupt_weights = guard.enforce(
        step=1, timestamp="2026-01-02", target_weights={"A": 0.5},
        current_positions={}, current_equity=0.0, is_bankrupt=True
    )
    assert all(w == 0.0 for w in bankrupt_weights.values()), "Bankrupt account must have all weights forced to 0.0"

    # 3. Test AccountingLedger non-negative cash & capital protection
    ledger = AccountingLedger(initial_cash=100.0, slippage_rate=0.0, commission_rate=0.0)
    
    # Attempt to buy 100 shares at $10 each ($1000 cost) with only $100 cash
    ledger.execute_pending_orders(
        target_weights={"XYZ": 1.0},
        open_prices={"XYZ": 10.0},
        base_equity=1000.0
    )
    assert ledger.cash >= 0.0, f"Cash balance must never be negative! Got: {ledger.cash}"
    assert ledger.positions.get("XYZ", 0.0) <= 10.0, "Cannot buy more shares than available cash allows"

    # 4. Test AccountingLedger mark-to-market bankruptcy floor at 0.0
    # Create adverse position
    ledger.positions["XYZ"] = 10.0
    ledger.cash = 0.0
    # Price crashes to 0 or negative
    close_equity = ledger.mark_to_market(close_prices={"XYZ": 0.0})
    assert close_equity == 0.0, f"Equity must be clamped to 0.0! Got {close_equity}"
    assert ledger.is_bankrupt, "Account must be marked as bankrupt"
    assert len(ledger.positions) == 0, "Positions must be cleared upon bankruptcy"
    assert ledger.cash == 0.0, "Cash must be 0.0 upon bankruptcy"

    print("  -> Passed.")


def main():
    print("\n" + "=" * 60)
    print("      INTER IIT TECH MEET 15.0 - SUBMISSION SANITY TEST")
    print("=" * 60 + "\n")
    try:
        test_file_structure()
        test_strategy_class()
        test_track_a_dummy()
        test_track_b_dummy()
        test_non_negative_capital_and_leverage()
        print("\n" + "=" * 60)
        print(" [SUCCESS] ALL PRE-SUBMISSION SANITY CHECKS PASSED!")
        print("=" * 60 + "\n")
        sys.exit(0)
    except Exception as e:
        print("\n" + "!" * 60)
        print(f" [FAILED] PRE-SUBMISSION SANITY CHECK ERROR: {e}")
        print("!" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
