# Inter IIT Tech Meet 15.0 Quant Selection

This document describes two candidate quantitative trading problem statements for **individual** team selection. Each participant chooses **one** of the two tracks and has uptil **8th October** to research, build, and validate a strategy independently.

If you submit both tasks, the one with the **higher score** would be considered

All strategies must be implemented against the **organizer-provided backtester**; building the backtester itself is **not** part of the task.

**Data fairness:** The dataset for *both* tracks is issued directly by the organizing committee at the start of the competition. Participants must not substitute, supplement, or source their own data for the assigned track --- this is to ensure that no participant gains an advantage from access to a better data vendor, and that all submissions are evaluated on research and modeling quality rather than data quality.

You may use any LLM to fast track the implementation of strategies. The goal is not to test your programming skills, but to test your research formulation and experimentation rigor.


Final selection will be based on a combination of 
- (a) The written research report
- (b) Code quality and correct backtester usage
- (c) Performance on an out-of-sample (OOS) held-out dataset not released to participants until after submission.
- (d) Interview for the participant to explain their strategy


Data : [Google Drive Link](https://drive.google.com/drive/folders/12wELmo8RhOMs6X-t59XTymcX8bfH5ZEC?usp=sharing)

# Track A: Volatility-Adjusted Cross-Sectional Strategy

## Statement
Given daily OHLCV data for a diversified universe of liquid assets, construct a cross-sectional strategy that ranks assets on a signal of your choosing (momentum, value, volatility, quality, etc.), sizes positions using **volatility-adjusted** weighting, and rebalances on a defined schedule. You are required to **statistically justify** both the ranking signal and the sizing methodology. Simply reporting a backtest Sharpe ratio is not sufficient.

## Data Provided
- **File Provided:** `Dataset_PS-A.csv` (In-sample period only for participants)
- **Universe:** 25 liquid large-cap style assets across multiple sectors.
- **Frequency:** Daily OHLCV
- **History:** 18 months total (12 months in-sample for development, 6 months OOS withheld by organizers for final evaluation).
- **Format:** Long-format consolidated CSV containing `date` (YYYY-MM-DD), `ticker` (asset identifier), `open`, `high`, `low`, `close` (adjusted), and `volume` (shares).

## Backtester
- **Release Date:** Within 5 days of competition start (by **9th September 2026**)
- **Documentation:** Complete usage guide, example workflows, API reference, and known edge cases will be provided
- **Q&A Window:** 48-hour support window upon release; questions via organizer forum
- **Integration:** All strategies must be validated using the **organizer-provided backtester only**; custom implementations will not be accepted
- **Key Features:** Automatic computation of Sharpe ratio, CAGR, Calmar ratio, maximum drawdown, walk-forward splits, and confidence intervals

## Required Deliverables
1. **Volatility Modeling:** Clear justification for the chosen vol-adjustment method in position sizing.

2. **Cross-Sectional Ranking:** Chosen ranking signal and economic rationale, rigorous statistical test of the ranking hypothesis, correlation/concentration analysis of top-ranked assets.

3. **Portfolio Construction:** Justified choice of optimization method against the data characteristics, well-defined rebalancing logic, and portfolio constraints.

4. **Validation:** Walk-forward backtesting with rigorous confidence intervals on key metrics, sensitivity analysis to key design parameters, and discussion of potential overfitting.

## Suggested Resources

**Volatility Modeling**
- Tsay – *Analysis of Financial Time Series*, Chapters 3–4
- Barndorff-Nielsen & Shephard (2004)
- Bollerslev (1986) – *Generalized Autoregressive Conditional Heteroskedasticity*
- Engle (1982) – *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation*

**Cross-Sectional Methods**
- Fama & MacBeth (1973)
- Cochrane – *Asset Pricing*, Chapter 8
- Ledoit & Wolf (2004)
- Asness, Moskowitz, and Pedersen (2013) – *Value and Momentum Everywhere*
- Ang, Hodrick, Xing, and Zhang (2006) – *The Cross-Section of Volatility and Expected Returns*

**Portfolio Construction**
- Markowitz (1952)
- Maillard et al. (2010) – Risk Parity
- DeMiguel et al. (2009)
- Rockafellar & Uryasev (2000) – CVaR
- Grinold & Kahn – *Active Portfolio Management* (Information Ratio & Fundamental Law)

**Validation**
- López de Prado – *Advances in Financial Machine Learning*, Chapter 6
- Efron & Tibshirani – *An Introduction to the Bootstrap*
- Harvey, Liu, and Zhu (2016) – *…and the Cross-Section of Expected Returns*

---

# Track B: Volatility Clustering & Flash Crash Detection (Micro-Regime Switching)

## Statement
Given second-by-second OHLCV data for a volatile intraday asset, build a strategy that detects micro-scale volatility regimes and adapts position sizing and exit rules accordingly. You must distinguish normal-volatility trading conditions from flash-crash/high-volatility regimes and reduce or exit exposure before severe drawdowns. All positions must be flat by end of day.

## Data Provided
- **Instrument:** High-Frequency Asset Microstructure data.
- **File Format:** Apache Parquet (`.parquet`).
- **Sampling Resolution:** Fixed 1 second ($\Delta t = 1$, 1-second klines).
- **Primary Index:** `tick` (Sequential integer starting from 0, timestamps stripped to prevent lookahead bias).
- **Columns:** `tick`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `count` (number of trades), `taker_buy_volume`.

## Backtester
- **Release Date:** Within 5 days of competition start (by **9th September 2026**)
- **Documentation:** Complete usage guide, example workflows, API reference, and sample Flash-Crash event data will be provided
- **Q&A Window:** 48-hour support window upon release; questions via organizer forum
- **Integration:** All strategies must be validated using the **organizer-provided backtester only**; custom implementations will not be accepted
- **Key Features:** Automatic computation of Sharpe ratio, CAGR, maximum drawdown, regime regime transition matrix, and walk-forward splits

## Microstructural Interpretation
- **Aggressive Order Flow Imbalance (OFI):** Measures buyer vs. seller pressure in a range of $[-1.0, +1.0]$: 
  `OFI = (taker_buy_volume - taker_sell_volume) / volume`
- **Volume Weighted Average Price (VWAP):** `quote_volume / volume`
- **Average Trade Size:** `volume / count`
- **Note:** Strict causal real-time processing is required. Scale invariance is necessary for OOS testing on unseen assets.

## Required Deliverables
1. **Volatility Analysis at Micro Scale:** Characterize volatility and its clustering behavior at second-level resolution. Discuss how your modeling approach informs regime detection.

2. **Regime Detection:** Develop a method to detect high-volatility or crash-prone regimes. Your model should demonstrate lead time—i.e., ability to flag elevated risk **before** severe price moves occur. Justify your approach and quantify its predictive power.

3. **Flash-Crash Characterization:** Identify and analyze high-volatility spike events in the data. Examine their precursors and dynamics. What microstructural signals (OFI, volume, momentum, volatility) are most informative?

4. **Strategy & Validation:** Build a strategy that adapts to detected regimes (e.g., position sizing adjusts to regime state). Validate via walk-forward backtesting. Compare regime-adapted performance vs. a naive baseline. All positions must be flat by end of day.

## Suggested Resources (Expanded)

**GARCH & Volatility**
- Tsay – *Analysis of Financial Time Series*, Chapters 3–4
- Barndorff-Nielsen & Shephard (2002, 2004)
- Andersen, Bollerslev, Diebold, and Labys (2001) – *The distribution of realized exchange rate volatility*

**Regime Switching & Flash Crashes**
- Hamilton (1989)
- Guidolin & Timmermann (2007)
- Kirilenko et al. (2017) – *The Flash Crash: High-Frequency Trading in an Electronic Market*
- Easley, de Prado, and O'Hara (2012) – *The Volume Clock: Insights into the High-Frequency Paradigm*
- Cont (2001) – *Empirical properties of asset returns: stylized facts and statistical issues*

**Microstructure & High Frequency**
- Cartea, Jaimungal, and Penalva (2015) – *Algorithmic and High-Frequency Trading*
- Avellaneda and Stoikov (2008) – *High-frequency trading in a limit order book*

---

# General Guidelines

- The organizer-provided backtester will be released within **5 days** of competition start, with full documentation and a 48-hour Q&A window. All strategies must use this backtester.
- No leverage (borrowing) is permitted.
- Both long and short positions are allowed.
- **Track A:** All positions close on rebalancing dates.
- **Track B:** All positions must be flat by end of day (strict EOD rule).
- The backtester will compute all relevant metrics (Sharpe, CAGR, Calmar, Drawdown, etc.) and generate plots.

# Common Evaluation Rubric

| Axis | Weight | What is graded |
| :--- | :--- | :--- |
| **Data Analysis** | 25% | Quality of EDA; correct handling of missing data, look-ahead bias; evidence-based volatility properties. |
| **Strategy Hypothesis / Model** | 35% | Rigorous testing of the signal hypothesis; statistical significance; soundness of regime detection / ranking methodology. |
| **Portfolio Optimization** | 25% | Justification of sizing/optimization method; correct implementation of constraints; sensible rebalancing logic. |
| **OOS Backtester Performance** | 15% | Performance on the organizer-held OOS dataset. Used as validation, but strong methodology scores well regardless. |


**Backtester Release:** Within 7 days of competition start (by Sep 11)  
**Final Deadline:** 8th October, 23:59 IST

# Submission Requirements
- **Source Code:** Python, using the provided backtester interface. Must be reproducible with fixed random seeds and no hardcoded absolute paths.
- **Written Report:** Max 10 pages covering hypothesis, methodology, validation, and results.
- **Dependencies:** `requirements.txt` with all required packages and versions.
- **Reproducibility:** Code must run on a fresh Python environment; random seeds must be fixed for deterministic output.