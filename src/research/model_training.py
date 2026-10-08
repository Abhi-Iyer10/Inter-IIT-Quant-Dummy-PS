"""
src/research/model_training.py - Ridge CV Factor Combination Optimization
Inter IIT Tech Meet 15.0 Quant Evaluation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit


class FactorModelTrainer:
    """Trains cross-validated regression models to determine optimal factor weights."""

    def __init__(self, mom_z: pd.DataFrame, rev_z: pd.DataFrame, prices: pd.DataFrame) -> None:
        self.mom_z = mom_z
        self.rev_z = rev_z
        self.fwd_rets = prices.pct_change(5).shift(-5)

    def optimize_factor_weights(self) -> dict:
        """Determines optimal weights w_mom and w_rev via RidgeCV on 5-day forward returns."""
        X_list, y_list = [], []
        common_dates = self.mom_z.dropna().index.intersection(self.fwd_rets.dropna().index)

        for date in common_dates:
            m_row = self.mom_z.loc[date]
            r_row = self.rev_z.loc[date]
            y_row = self.fwd_rets.loc[date]
            df_temp = pd.DataFrame({"mom": m_row, "rev": r_row, "target": y_row}).dropna()
            
            if not df_temp.empty:
                X_list.append(df_temp[["mom", "rev"]].values)
                y_list.append(df_temp["target"].values)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        tscv = TimeSeriesSplit(n_splits=5)
        model = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=tscv)
        model.fit(X, y)

        raw_coefs = model.coef_
        normalized_weights = raw_coefs / np.sum(np.abs(raw_coefs))
        return {
            "raw_mom_weight": float(raw_coefs[0]),
            "raw_rev_weight": float(raw_coefs[1]),
            "norm_w_mom": float(normalized_weights[0]),
            "norm_w_rev": float(normalized_weights[1]),
            "alpha": float(model.alpha_)
        }