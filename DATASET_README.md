# Track A: Volatility-Adjusted Cross-Sectional Strategy

## Overview

Participants must design a **cross-sectional** trading strategy that:
1. Ranks a universe of liquid assets on a signal of their choice (momentum, value, volatility, quality, etc.).
2. Sizes positions using **volatility-adjusted** weighting.
3. Rebalances on a defined schedule.

You are required to **statistically justify** both the ranking signal and the sizing methodology. Simply reporting a backtest Sharpe ratio is not sufficient.

---

## Data

### File Provided to Participants
- `Dataset_PS-A.csv` — In-sample (IS) period only

### Format
Long-format consolidated CSV:

| Column  | Description                          | Type    |
|---------|--------------------------------------|---------|
| date    | Trading date (YYYY-MM-DD)            | string  |
| ticker  | Asset identifier                     | string  |
| open    | Opening price                        | float   |
| high    | Highest price                        | float   |
| low     | Lowest price                         | float   |
| close   | Closing price (adjusted)             | float   |
| volume  | Trading volume (shares)              | integer |

- Frequency: Daily
- Universe size: 24 liquid large-cap style assets across multiple sectors


---

## Required Deliverables

### 1. Volatility Modeling
- Clear justification for the chosen volatility-adjustment method used in position sizing

### 2. Cross-Sectional Ranking
- Chosen ranking signal and economic rationale
- Statistical test of the ranking hypothesis
- Correlation / concentration analysis of top-ranked assets

### 3. Portfolio Construction
- Optimization / weighting method
- Justification against the data characteristics
- Rebalancing logic
- Position limits and sector limits (if any)

### 4. Validation
- Walk-forward backtesting (rolling train/test windows)
- Confidence intervals on Sharpe ratio (bootstrap or other rigorous method)
- Sensitivity analysis to key parameters

---

## Suggested Resources

**Volatility Modeling**
- Tsay – *Analysis of Financial Time Series*, Chapters 3–4
- Barndorff-Nielsen & Shephard (2004)

**Cross-Sectional Methods**
- Fama & MacBeth (1973)
- Cochrane – *Asset Pricing*, Chapter 8
- Ledoit & Wolf (2004)

**Portfolio Construction**
- Markowitz (1952)
- Maillard et al. (2010) – Risk Parity
- DeMiguel et al. (2009)
- Rockafellar & Uryasev (2000) – CVaR

**Validation**
- López de Prado – *Advances in Financial Machine Learning*, Chapter 6
- Efron & Tibshirani – *An Introduction to the Bootstrap*

---

## Submission Guidelines

Participants should submit:
1. A written report (PDF) covering all four required deliverables.
2. Clean, reproducible code (Python) that can be run end-to-end on the provided `Dataset_PS-A.csv`.
3. A clear description of the final strategy that will be evaluated on the private OOS period.

---

## Important Notes

- Position sizing must be volatility-aware.

Good luck. Focus on **justification and robustness**, not just in-sample performance.

# Dataset Documentation: High-Frequency Asset Microstructure (1s Granularity)

## Overview
This dataset contains anonymized, high-frequency market data sampled at **1-second intervals**. It is engineered specifically for quantitative research, high-frequency volatility modeling, regime detection (e.g., Hidden Markov Models), and flash crash detection algorithms.

## Dataset Specifications
* **File Format:** Apache Parquet (`.parquet`)
* **Sampling Resolution:** Fixed 1 second ($\Delta t = 1$)
* **Primary Index:** `tick` (Sequential integer starting from `0`)
* **Price/Volume Scaling:** Raw market units (Unscaled to preserve true market dynamics)

---

## Field Definitions

| Column Name | Data Type | Measurement Unit | Description |
| :--- | :--- | :--- | :--- |
| **`tick`** *(Index)* | `Int64` | Discrete Seconds | Zero-based sequential time index representing elapsed seconds ($0, 1, 2, \dots, N$). |
| **`open`** | `Float64` | Quote Currency | Price of the first trade executed during the 1-second window. |
| **`high`** | `Float64` | Quote Currency | Highest price executed during the 1-second window. |
| **`low`** | `Float64` | Quote Currency | Lowest price executed during the 1-second window. |
| **`close`** | `Float64` | Quote Currency | Price of the final trade executed during the 1-second window. |
| **`volume`** | `Float64` | Base Asset | Total quantity of base asset traded within the 1-second window. |
| **`quote_volume`** | `Float64` | Quote Currency | Total monetary value traded ($\sum \text{Price} \times \text{Volume}$) within the 1-second window. |
| **`count`** | `Int64` | Trades | Total count of distinct matched order transactions filled within the second. |
| **`taker_buy_volume`** | `Float64` | Base Asset | Total volume executed by aggressive buyers using market buy orders. |

---

## Microstructural Interpretation & Quant Metrics

### 1. Time Continuity
* Timestamps have been stripped to prevent calendar lookups and historical backtest fitting.
* Because data is sampled at uniform 1-second ticks, you can treat $\Delta t = 1$ step for discrete time-series models.

### 2. Order Flow & Net Aggressiveness
In high-frequency limit order books, every trade requires a passive **Maker** (limit order) and an aggressive **Taker** (market order).

* **Aggressive Buy Volume:** Provided directly by `taker_buy_volume`.
* **Aggressive Sell Volume:** Derived by subtracting taker buy volume from total volume:
  $$\text{Taker Sell Volume} = \text{volume} - \text{taker\_buy\_volume}$$
* **Order Flow Imbalance (OFI):** Measures buyer vs. seller pressure in a range of $[-1.0, +1.0]$:
  $$\text{OFI} = \frac{\text{taker\_buy\_volume} - \text{taker\_sell\_volume}}{\text{volume}}$$

### 3. Liquidity & Price Impact
* **Volume Weighted Average Price (VWAP):** Represents the true average execution price in that second:
  $$\text{VWAP} = \frac{\text{quote\_volume}}{\text{volume}}$$
* **Average Trade Size:** Indicates retail vs. institutional order size:
  $$\text{Average Trade Size} = \frac{\text{volume}}{\text{count}}$$
* **Intraday Volatility Proxy (Log Range):**
  $$\sigma_{\text{prox}} = \ln(\text{high}) - \ln(\text{low})$$

---

# Important Notes

1. **Strict Causal Real-Time Processing:** Models/Strategies must be strictly forward-looking. Feature calculations (rolling Z-scores, moving averages, standard deviations) must only use historical inputs ($t-k \le t$). Global normalization methods (e.g., fitting a `StandardScaler` over the entire dataset) are strictly prohibited.
2. **Scale Invariance:** Models will be tested Out-of-Sample (OOS) on unseen assets with different price and volume regimes. Any engineered features must be dimensionless or scale-free (e.g., ratios, returns, Z-scores) to ensure model parameters generalize.
