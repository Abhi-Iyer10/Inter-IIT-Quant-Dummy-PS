"""
src/research/statistical_analysis.py - Backtest Risk Analytics & DSR Evaluation
Inter IIT Tech Meet 15.0 Quant Evaluation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalValidator:
    """Computes Sharpe, Sortino, Drawdown, and Deflated Sharpe Ratio (DSR)."""

    @staticmethod
    def calculate_dsr(
        estimated_sharpe: float,
        num_trials: int = 15,
        backtest_length: int = 1484,
        skew: float = -0.12,
        kurtosis: float = 3.45
    ) -> float:
        """
        Calculates Probabilistic / Deflated Sharpe Ratio (DSR) under trial variance.
        Formula accounts for non-Gaussian return distributions and N trial backtests.
        """
        sr_std = np.sqrt((1.0 + (0.5 * estimated_sharpe ** 2) - (skew * estimated_sharpe) + 
                          ((kurtosis - 3) / 4.0) * estimated_sharpe ** 2) / (backtest_length - 1))
        
        # Expected max Sharpe under N independent trials
        euler_mascheroni = 0.5772156649
        exp_max_sr = sr_std * ((1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / num_trials) + 
                                euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (num_trials * np.e)))
        
        dsr_z = (estimated_sharpe - exp_max_sr) / sr_std
        p_dsr = float(1.0 - stats.norm.cdf(dsr_z))
        return p_dsr