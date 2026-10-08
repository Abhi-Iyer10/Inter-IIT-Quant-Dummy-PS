"""
src/research/factor_research.py - Alpha Factor Construction & IC Analysis
Inter IIT Tech Meet 15.0 Quant Evaluation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


class FactorResearcher:
    """Cross-sectional alpha factor generator and Information Coefficient (IC) evaluator."""

    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices.copy()

    def generate_momentum(self, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
        """12m - 1m Cross-Sectional Momentum factor (skipping recent 21 days)."""
        return (self.prices.shift(skip) - self.prices.shift(lookback)) / self.prices.shift(lookback)

    def generate_reversal(self, skip: int = 21) -> pd.DataFrame:
        """1m Short-Term Reversal factor (-1 * 21-day return)."""
        return -1.0 * ((self.prices - self.prices.shift(skip)) / self.prices.shift(skip))

    @staticmethod
    def zscore_normalize(factor_df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional Z-score standardization."""
        mean = factor_df.mean(axis=1)
        std = factor_df.std(axis=1)
        return factor_df.sub(mean, axis=0).div(std + 1e-8, axis=0)

    def evaluate_ic(self, factor_df: pd.DataFrame, forward_horizon: int = 5) -> pd.Series:
        """Computes time-series Pearson and Spearman Rank IC metrics."""
        fwd_returns = self.prices.pct_change(forward_horizon).shift(-forward_horizon)
        rank_ics = []

        for date in factor_df.index:
            f_row = factor_df.loc[date].dropna()
            r_row = fwd_returns.loc[date].dropna()
            common = f_row.index.intersection(r_row.index)

            if len(common) >= 5:
                ic, _ = stats.spearmanr(f_row[common], r_row[common])
                if not np.isnan(ic):
                    rank_ics.append(ic)

        arr = np.array(rank_ics)
        t_stat, p_val = stats.ttest_1samp(arr, 0.0)
        return pd.Series({
            "mean_ic": np.mean(arr),
            "std_ic": np.std(arr),
            "ic_ir": np.mean(arr) / (np.std(arr) + 1e-8),
            "t_statistic": t_stat,
            "p_value": p_val
        })