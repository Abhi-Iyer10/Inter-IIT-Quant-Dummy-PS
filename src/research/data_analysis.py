"""
src/research/data_analysis.py - Exploratory Data Analysis & Diagnostic Utilities
Inter IIT Tech Meet 15.0 Quant Evaluation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


class DataAnalyzer:
    """Exploratory data analysis and time-series diagnostic utilities."""

    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices.copy().dropna(how="all")
        self.returns = self.prices.pct_change().dropna()

    def test_stationarity(self) -> pd.DataFrame:
        """Performs ADF tests to verify log-return stationarity."""
        results = []
        for col in self.returns.columns:
            res = adfuller(self.returns[col].dropna())
            results.append({"ticker": col, "adf_stat": res[0], "p_value": res[1]})
        return pd.DataFrame(results).set_index("ticker")

    def evaluate_volatility_window_stability(self) -> pd.DataFrame:
        """Compares realized volatility stability across 10d, 30d, and 60d windows."""
        log_rets = np.log(self.prices / self.prices.shift(1))
        vol_10d = log_rets.rolling(10).std() * np.sqrt(252)
        vol_30d = log_rets.rolling(30).std() * np.sqrt(252)
        vol_60d = log_rets.rolling(60).std() * np.sqrt(252)

        return pd.DataFrame({
            "vol_10d_autocorr": [vol_10d[col].autocorr(1) for col in vol_10d.columns],
            "vol_30d_autocorr": [vol_30d[col].autocorr(1) for col in vol_30d.columns],
            "vol_60d_autocorr": [vol_60d[col].autocorr(1) for col in vol_60d.columns]
        })