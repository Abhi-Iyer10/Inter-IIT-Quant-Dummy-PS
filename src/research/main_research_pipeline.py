"""
src/research/main_research_pipeline.py - End-to-End Parameter Search Pipeline
Inter IIT Tech Meet 15.0 Quant Evaluation
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from data_analysis import DataAnalyzer
from factor_research import FactorResearcher
from model_training import FactorModelTrainer
from statistical_analysis import StatisticalValidator


def execute_parameter_search(prices_df: pd.DataFrame) -> None:
    """Executes research workflow to derive Track A parameters reported in submission."""
    # Step 1: Volatility stability check
    analyzer = DataAnalyzer(prices_df)
    vol_stability = analyzer.evaluate_volatility_window_stability()
    print("1. Volatility Window Autocorrelation (30d selected):")
    print(vol_stability.mean())

    # Step 2: Factor IC Evaluation
    researcher = FactorResearcher(prices_df)
    mom_z = researcher.zscore_normalize(researcher.generate_momentum(252, 21))
    rev_z = researcher.zscore_normalize(researcher.generate_reversal(21))
    
    ic_stats = researcher.evaluate_ic(mom_z, forward_horizon=5)
    print("\n2. Momentum Factor Rank IC Results:")
    print(ic_stats)

    # Step 3: Weight Optimization
    trainer = FactorModelTrainer(mom_z, rev_z, prices_df)
    opt_weights = trainer.optimize_factor_weights()
    print("\n3. RidgeCV Normalized Factor Weights:")
    print(f"   w_mom: {opt_weights['norm_w_mom']:.2f}, w_rev: {opt_weights['norm_w_rev']:.2f}")

    # Step 4: Overfitting & DSR Validation
    p_dsr = StatisticalValidator.calculate_dsr(estimated_sharpe=0.81, num_trials=15, backtest_length=1484)
    print(f"\n4. Deflated Sharpe Ratio (DSR) p-value: {p_dsr:.4f} (< 0.05 Target)")


if __name__ == "__main__":
    import pandas as pd

    # Load dataset
    raw_df = pd.read_csv("data/Dataset_PS-A.csv")

    # Pivot long format to wide matrix
    if "ticker" in raw_df.columns and "close" in raw_df.columns:
        prices_df = raw_df.pivot(index="date", columns="ticker", values="close")
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index().ffill()
    else:
        prices_df = pd.read_csv("data/Dataset_PS-A_2.csv", index_col=0, parse_dates=True).sort_index().ffill()

    print(f"Loaded prices matrix with shape: {prices_df.shape} (Dates x Tickers)")

    # Execute optimization pipeline
    execute_parameter_search(prices_df)