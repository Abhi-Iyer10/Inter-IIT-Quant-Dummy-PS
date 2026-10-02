# Quantitative Backtester User & Developer Guide

**Inter IIT Tech Meet 15.0 — Quantitative Trading Evaluation Framework**

Welcome to the official backtesting engine for the Inter IIT Tech Meet 15.0 Quant Selection. This guide provides comprehensive documentation on how to configure, develop, debug, and submit your quantitative trading strategies for **Track A** and **Track B**.

---

## Table of Contents

- [Quantitative Backtester User \& Developer Guide](#quantitative-backtester-user--developer-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Overview \& Architectural Philosophy](#1-overview--architectural-philosophy)
  - [2. Quickstart (Run in 60 Seconds)](#2-quickstart-run-in-60-seconds)
    - [Step 0: Fork the repository and clone it on your machine](#step-0-fork-the-repository-and-clone-it-on-your-machine)
    - [Step 1: Set Up Environment](#step-1-set-up-environment)
    - [Step 2: Run Track A (Daily Cross-Sectional)](#step-2-run-track-a-daily-cross-sectional)
    - [Step 3: Run Track B (1-Second Microstructure)](#step-3-run-track-b-1-second-microstructure)
    - [Step 4: Run Pre-Submission Sanity Check](#step-4-run-pre-submission-sanity-check)
  - [3. Repository Layout \& Code Boundaries](#3-repository-layout--code-boundaries)
  - [4. How to Structure Your Strategy Code](#4-how-to-structure-your-strategy-code)
    - [Mandatory Entry Point: `src/engine.py`](#mandatory-entry-point-srcenginepy)
    - [Lifecycle: `initialize(context)`](#lifecycle-initializecontext)
    - [Lifecycle: `on_bar(context, bars)`](#lifecycle-on_barcontext-bars)
      - [How to detect the active track:](#how-to-detect-the-active-track)
      - [How to submit orders:](#how-to-submit-orders)
    - [Recommended Directory Modularity](#recommended-directory-modularity)
    - [High-Frequency Optimization Rules (Track B)](#high-frequency-optimization-rules-track-b)
  - [5. Core API Contracts (`strategy_base.py`)](#5-core-api-contracts-strategy_basepy)
    - [The `Context` Object](#the-context-object)
    - [The `Bar` \& `FastBarView` Data Structures](#the-bar--fastbarview-data-structures)
      - [Direct OHLCV Attributes:](#direct-ohlcv-attributes)
      - [Microstructure Properties (Cached \& Computed Lazily):](#microstructure-properties-cached--computed-lazily)
  - [6. Execution Simulation \& Order Mechanics](#6-execution-simulation--order-mechanics)
    - [Zero Lookahead Bias (Next-Bar Open Execution)](#zero-lookahead-bias-next-bar-open-execution)
    - [Target Allocation \& Share Rebalancing Math](#target-allocation--share-rebalancing-math)
    - [Transaction Friction Models (Slippage \& Commission)](#transaction-friction-models-slippage--commission)
    - [Mark-to-Market Valuation](#mark-to-market-valuation)
  - [7. Hard Constraints \& Rule Guard](#7-hard-constraints--rule-guard)
    - [Gross Leverage Limit ($\\le 1.0$)](#gross-leverage-limit-le-10)
    - [Strict Non-Negative Capital Constraint ($\\text{Capital} \\ge 0$)](#strict-non-negative-capital-constraint-textcapital-ge-0)
  - [8. Configuration \& CLI Execution](#8-configuration--cli-execution)
    - [Configuring `config.yaml`](#configuring-configyaml)
    - [Command-Line Flags (`run_backtest.py`)](#command-line-flags-run_backtestpy)
  - [9. Interpreting Results \& Diagnostic Artifacts](#9-interpreting-results--diagnostic-artifacts)
    - [Evaluation Metrics Table](#evaluation-metrics-table)
      - [Key Diagnostic Insights:](#key-diagnostic-insights)
    - [`results/summary.json`](#resultssummaryjson)
    - [`results/trades.csv`](#resultstradescsv)
    - [4-Panel Performance Tearsheet (`results/tearsheet.png`)](#4-panel-performance-tearsheet-resultstearsheetpng)
  - [10. Pre-Submission Sanity Checks \& Best Practices](#10-pre-submission-sanity-checks--best-practices)
    - [Common Pitfalls to Avoid:](#common-pitfalls-to-avoid)

---

## 1. Overview & Architectural Philosophy

The backtester is designed to evaluate quantitative trading strategies under realistic market conditions:

- **Zero Lookahead Bias:** The engine operates in a strict event-driven loop. Signals generated at step $t$ are executed at the Open price of step $t+1$.
- **Unified API Contract:** Both Track A (multi-asset daily cross-sectional) and Track B (1-second high-frequency microstructure) use the exact same strategy interface (`BaseStrategy`, `Context`, `Bar`).
- **High-Throughput Streaming Engine:** Using contiguous NumPy buffers and flyweight bar views, the engine processes 3.88M ticks for Track B in under 25 seconds (>170,000 ticks/sec) in pure Python.
- **Strict Risk Enforcement:** Zero margin borrowing ($\sum |w_i| \le 1.0$) and a strict non-negative capital constraint are enforced deterministically on every single simulation step.

---

## 2. Quickstart (Run in 60 Seconds)

### Step 0: Fork the repository and clone it on your machine

1. On GitHub, open the repository and select **Fork** to create a copy under your account.
2. Clone your fork and enter the project directory:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Replace `<your-username>` and `<repository-name>` with your GitHub username and the repository name. Run the following steps from this directory.

### Step 1: Set Up Environment

Ensure you have Python 3.14+ installed:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Run Track A (Daily Cross-Sectional)

```bash
python3 run_backtest.py --track A
```

### Step 3: Run Track B (1-Second Microstructure)

For rapid local iteration, test against a slice of ticks first (e.g., 1 day = 86,400 ticks):

```bash
# Fast test: 1 day of streaming data
python3 run_backtest.py --track B --max-ticks 86400

# Full backtest: all 3.88M ticks
python3 run_backtest.py --track B
```

### Step 4: Run Pre-Submission Sanity Check

Before submitting your repository, verify that your code satisfies all structural and risk constraints:

```bash
python3 test_submission.py
```

---

## 3. Repository Layout & Code Boundaries

The repository strictly separates the **Participant Space** (where you write your strategy) from the **Harness/Organizer Space** (the trusted evaluation engine):

```text
Backtester/
├── BACKTESTER_README.md      # THIS MANUAL: Complete participant usage guide
├── README.md                 # System architecture & specification blueprint
├── DATASET_README.md         # Detailed dataset schemas and field specifications
├── requirements.txt          # [PARTICIPANT SPACE] Used libraries : Some initial libraries may be there, you can add more if you are using them
├── config.yaml               # Participant parameters (lookback, cadence, tracks)
│
├── strategy_base.py          # [DO NOT MODIFY] Core API contracts (BaseStrategy, Context, Bar)
├── backtester.py             # [DO NOT MODIFY] Engine core, ledger, rule guard, metrics
├── run_backtest.py           # Command-line entry runner
├── test_submission.py        # Automated pre-submission validation script
│
├── data/                     # In-Sample development datasets
│   ├── Dataset_PS-A.csv      # Track A daily cross-sectional data (24 assets)
│   └── ASSET_ALPHA_1s.parquet# Track B 1-second continuous microstructure ticks
│
├── results/                  # Auto-generated outputs (created upon running backtest)
│   ├── summary.json          # Machine-readable evaluation metrics
│   ├── trades.csv            # Tick-by-tick order fill logs for debugging
│   └── tearsheet.png         # 4-panel visual performance charts
│
└── src/                      # [PARTICIPANT SPACE] Your Strategy Code
    ├── __init__.py
    ├── engine.py             # MANDATORY ENTRY POINT: Contains ParticipantStrategy
    ├── models/               # Your statistical/ML models (GARCH, Kelly, Regime, Regression, etc.)
    ├── utils/                # Your feature extraction, math, and data helpers
    ├── research/             # Research, data analysis, and model-training work
    └── report.pdf            # Strategy rationale, analysis, and approach (max. 10 pages)
```

> [!WARNING]
> **Do not modify `strategy_base.py`, `backtester.py`, or `test_submission.py`.**
> When your submission is graded on the held-out Out-Of-Sample (OOS) dataset, the organizer's clean, canonical evaluation engine will be used. Only files inside `src/` and `config.yaml` will be evaluated.

---

## 4. How to Structure Your Strategy Code

### Mandatory Entry Point: `src/engine.py`

The backtester imports your strategy by loading `ParticipantStrategy` from `src/engine.py`. Your strategy class **must** inherit from `BaseStrategy` and implement `initialize()` and `on_bar()`:

```python
# src/engine.py
from strategy_base import BaseStrategy, Context, Bar
from typing import Dict

class ParticipantStrategy(BaseStrategy):
    def initialize(self, context: Context) -> None:
        """Called once before the backtest begins."""
        pass

    def on_bar(self, context: Context, bars: Dict[str, Bar]) -> None:
        """Called at every simulation step t."""
        pass
```

---

### Lifecycle: `initialize(context)`

`initialize` is executed **once** at step 0 before any market bars are processed. Use this method to:

- Configure hyperparameters (lookback periods, rebalance intervals).
- Inspect universe metadata (`context.universe` or `context.symbol`).
- Allocate rolling memory buffers (lists, deques, or pre-allocated NumPy arrays).
- Instantiate and initialize statistical models, estimators, or pre-trained model weights.

```python
def initialize(self, context: Context) -> None:
    # Read universe
    self.universe = context.universe
    self.rebalance_cadence = 5
    self.history = {ticker: [] for ticker in self.universe}
  
    # Store custom objects in context scratchpad if needed
    context.extra["model_ready"] = True
```

---

### Lifecycle: `on_bar(context, bars)`

`on_bar` is called at every simulation time step $t$.

- `context`: Provides current marked-to-market portfolio value, available cash balance, active holdings, and target weight setters.
- `bars`: A dictionary mapping `ticker -> Bar` object for step $t$.
  - In **Track A**, `bars` contains all assets in the universe for that trading day.
  - In **Track B**, `bars` contains a single entry for the active asset (e.g. `ALPHA`).

#### How to detect the active track:

```python
def on_bar(self, context: Context, bars: Dict[str, Bar]) -> None:
    if len(bars) > 1:
        self._handle_track_a(context, bars)
    else:
        self._handle_track_b(context, bars)
```

#### How to submit orders:

In either track, you express decisions by setting **target portfolio weights** $w_i \in [-1.0, 1.0]$:

```python
# Multi-asset mode (Track A):
target_weights = {"ASSET_00": 0.20, "ASSET_01": -0.15, "ASSET_02": 0.10}
context.set_target_weights(target_weights)

# Single-asset mode (Track B):
context.set_target_weight(0.50)  # Allocate 50% of portfolio equity
```

---

### Recommended Directory Modularity

Keep `src/engine.py` concise and maintainable by placing specialized logic in sub-packages. An example is given below:

- **`src/models/`**:
  - `volatility.py`: Volatility Engine API.
  - `regime.py`: Regime Detection API.
  - `kelly.py`: Kelly criterion for portfolio optimisation.
- **`src/utils/`**:
  - `microstructure.py`: proxy estimators API.
  - `math_helpers.py`: Fast exponential smoothers, rolling statistics, and bootstrap confidence intervals API.
- **`src/research/`**:
  - `data-analysis.ipynb`: Data analysis, distribution analysis etc.
  - `model-training.ipynb`: Data preparation, model training, cross validation etc.
- **`src/report.pdf`**:
  - A report describing the approach.

---

### High-Frequency Optimization Rules (Track B)

Track B processes **3,888,000 ticks**. Naive implementations will cause severe backtest slowdowns. Follow these performance rules:

1. **Never create Pandas DataFrames inside `on_bar()`:** Creating a DataFrame every tick takes ~500 microseconds. Across 3.88M ticks, that translates to over 30 minutes of runtime!
2. **Use Scalar Math or Fixed Deques:** Maintain rolling windows using `collections.deque(maxlen=N)` or fixed-size NumPy circular buffers.
3. **Decouple Prediction from Execution:** If your time series model (e.g., GARCH or Kalman Filter) is computationally intensive, update the model every $K$ ticks (e.g., every 5, 10, or 60 seconds) rather than at every single second tick.
4. **Use Lazy Properties:** Access `bar.ofi`, `bar.vwap`, and `bar.log_range` only when your logic requires them.

---

## 5. Core API Contracts (`strategy_base.py`)

### The `Context` Object

Passed into both `initialize()` and `on_bar()`:

| Property / Method                       | Type                    | Description                                                                                     |
| :-------------------------------------- | :---------------------- | :---------------------------------------------------------------------------------------------- |
| `context.portfolio_value`             | `float`               | Current marked-to-market equity:$\text{Cash} + \sum (\text{Shares}_i \times \text{Close}_i)$. |
| `context.cash`                        | `float`               | Current unallocated cash balance.                                                               |
| `context.initial_cash`                | `float`               | Starting balance ($100,000.00).                                                                 |
| `context.step`                        | `int`                 | Current integer step index ($0, 1, 2, \dots$).                                                |
| `context.timestamp`                   | `Any`                 | Current timestamp (date string for Track A, integer tick for Track B).                          |
| `context.universe`                    | `List[str]`           | List of all asset identifiers in the active track.                                              |
| `context.symbol`                      | `Optional[str]`       | Active ticker symbol for Track B (e.g.,`"ALPHA"`).                                            |
| `context.positions`                   | `Mapping[str, float]` | Read-only dictionary of current open shares:`{ticker: quantity}`.                             |
| `context.extra`                       | `Dict[str, Any]`      | User scratchpad dictionary for storing state across bars.                                       |
| `context.set_target_weights(weights)` | `Method`              | Queues target portfolio allocations$\{ \text{ticker}: w_i \}$.                                |
| `context.set_target_weight(weight)`   | `Method`              | Convenience helper for Track B (sets allocation for`context.symbol`).                         |

---

### The `Bar` & `FastBarView` Data Structures

Represents market data for an asset at time step $t$.

#### Direct OHLCV Attributes:

- `bar.ticker`: Asset symbol string.
- `bar.timestamp`: Simulation timestamp (date string in Track A, integer tick in Track B).
- `bar.open`: Opening price of the bar.
- `bar.high`: Highest price during the bar.
- `bar.low`: Lowest price during the bar.
- `bar.close`: Closing price of the bar.
- `bar.volume`: Traded volume (shares or base units).
- `bar.quote_volume`: Total traded value ($\sum P \times V$) [Track B].
- `bar.count`: Number of matched trades during the second [Track B].
- `bar.taker_buy_volume`: Aggressive buyer volume executed at ask [Track B].

#### Microstructure Properties (Cached & Computed Lazily):

- **`bar.ofi` (Order Flow Imbalance):**

  $$
  \text{OFI} = \frac{\text{taker\_buy\_volume} - \text{taker\_sell\_volume}}{\text{volume}} \in [-1.0, 1.0]
  $$

  Quantifies aggressive buyer pressure ($>0$) vs. aggressive seller pressure ($<0$).
- **`bar.vwap`:** Volume-Weighted Average Price ($\frac{\text{quote\_volume}}{\text{volume}}$).
- **`bar.log_range`:** Volatility proxy $\ln(\text{High}) - \ln(\text{Low})$.
- **`bar.taker_sell_volume`:** Passive buy volume filled by market sells ($\text{volume} - \text{taker\_buy\_volume}$).

---

## 6. Execution Simulation & Order Mechanics

### Zero Lookahead Bias (Next-Bar Open Execution)

The backtester eliminates lookahead bias via asynchronous next-bar execution:

```
Step t:
  1. Open(t) arrives -> Execute pending orders submitted at step t-1
  2. Close(t) arrives -> Mark portfolio equity to market
  3. on_bar(context, Bar(t)) called -> Strategy analyzes bar t and queues target weights w(t)

Step t+1:
  1. Open(t+1) arrives -> Execute pending target weights w(t) at Open(t+1)
```

Orders generated from bar $t$ are filled at the **Open price of bar $t+1$**. You never execute at the Close price of the same bar you used to generate the signal.

---

### Target Allocation & Share Rebalancing Math

When you specify target weight $w_i \in [-1.0, 1.0]$, the required share adjustment is:

$$
\Delta \text{Shares}_i = \frac{\text{Equity}_t \cdot w_i}{\text{Open}_{i, t+1}} - \text{Shares}_{i, t}
$$

- If $\Delta \text{Shares}_i > 0$: An order to **BUY** $\Delta \text{Shares}_i$ is executed.
- If $\Delta \text{Shares}_i < 0$: An order to **SELL** $|\Delta \text{Shares}_i|$ is executed.
- If $\Delta \text{Shares}_i = 0$: No order is generated.

---

### Transaction Friction Models (Slippage & Commission)

To prevent unrealistic high-frequency scalping, every order pays realistic friction:

1. **Execution Slippage:**

   - Buy Orders: $\quad P_{\text{fill}} = \text{Open}_{t+1} \times (1 + \text{SlippageRate})$
   - Sell Orders: $\quad P_{\text{fill}} = \text{Open}_{t+1} \times (1 - \text{SlippageRate})$
     *(Default: 0.00% = 0 bps)*
2. **Transaction Commission:**

   $$
   \text{Commission} = |\Delta \text{Shares}| \times P_{\text{fill}} \times \text{CommissionRate}
   $$

   *(Default: 0.01% = 1 bp)*

---

### Mark-to-Market Valuation

At the close of each step $t$:

$$
\text{Equity}_t = \text{Cash}_t + \sum_{i=1}^N \text{Shares}_{i, t} \times \text{Close}_{i, t}
$$

---

## 7. Hard Constraints & Rule Guard

The backtester includes an automated **`RuleGuard`** and strict accounting constraints. Submissions that repeatedly violate these rules will be penalized or disqualified.

### Gross Leverage Limit ($\le 1.0$)

- **The Rule:** Total absolute portfolio exposure must never exceed 100% of equity:
  $$
  \text{Gross Leverage} = \sum_{i=1}^N |w_i| \le 1.0
  $$
- **Automatic Clamping:** If $\sum |w_i| > 1.0$, the engine logs a **Leverage Violation** and rescales allocations:
  $$
  w_i^{\text{clamped}} = \frac{w_i}{\sum_{k} |w_k|}
  $$
- **Best Practice:** Always normalize your raw weights before calling `set_target_weights`:
  ```python
  total_weight = sum(abs(w) for w in raw_weights.values())
  if total_weight > 1.0:
      weights = {k: v / total_weight for k, v in raw_weights.items()}
  ```

---

### Strict Non-Negative Capital Constraint ($\text{Capital} \ge 0$)

- **The Rule:** The backtester does not provide margin loans. Your available cash and marked-to-market equity must remain non-negative at all times.
- **Cash-Bounded Order Sizing:** Buy orders are capped by available cash balance. The ledger will never allow a buy order to drive cash below $0.00.
- **Bankruptcy Halt:** If marked-to-market equity drops to $\le 0.00$, the account is marked bankrupt:
  - All open positions are immediately liquidated to 0.
  - Cash and equity are set to $0.00$.
  - Trading is permanently terminated for the remainder of the backtest.
  - Returns and Sharpe ratio drop to failure levels.

---

## 8. Configuration & CLI Execution

### Configuring `config.yaml`

Central settings can be adjusted in `config.yaml`:

```yaml
track: "A"                  # Default track: "A" or "B"
initial_cash: 100000.0      # Starting capital ($100,000.00)
slippage_rate: 0.0000       # 0 bps execution slippage
commission_rate: 0.0001     # 1 bp commission

track_a:
  data_path: "data/Dataset_PS-A.csv"
  rebalance_cadence: 5      # Days between rebalances

track_b:
  data_path: "data/ASSET_ALPHA_1s.parquet"
  symbol: "ALPHA"
  max_ticks: null           # null = all 3.88M ticks; or set integer for quick tests

output_dir: "results"       # Directory for summary.json, trades.csv, tearsheet.png
generate_plot: true         # Generate matplotlib tearsheet
```

---

### Command-Line Flags (`run_backtest.py`)

All parameters can be overridden from the command line:

| Flag               | Argument       | Description                        | Example                                                |
| :----------------- | :------------- | :--------------------------------- | :----------------------------------------------------- |
| `--track`        | `A` or `B` | Problem track to evaluate          | `python run_backtest.py --track A`                   |
| `--max-ticks`    | `int`        | Limit number of ticks for Track B  | `python run_backtest.py --track B --max-ticks 86400` |
| `--config`       | `filepath`   | Path to custom YAML configuration  | `python run_backtest.py --config my_config.yaml`     |
| `--data`         | `filepath`   | Override path to dataset file      | `python run_backtest.py --data custom_data.parquet`  |
| `--initial-cash` | `float`      | Override starting cash             | `python run_backtest.py --initial-cash 50000`        |
| `--output-dir`   | `directory`  | Directory for artifacts            | `python run_backtest.py --output-dir my_results`     |
| `--no-plot`      | *flag*       | Disable matplotlib plot generation | `python run_backtest.py --track B --no-plot`         |

---

## 9. Interpreting Results & Diagnostic Artifacts

### Evaluation Metrics Table

When execution completes, a structured evaluation table is printed to your terminal:

```text
=================================================================
                QUANT BACKTEST EVALUATION SUMMARY          
=================================================================
-- PORTFOLIO PERFORMANCE -----------------------------------------
  Track                              :                        B
  Total Steps                        :                   86,400
  Execution Duration (s)             :                     0.49
  Throughput (ticks/s)               :                  174,677
  Initial Cash                       :               100,000.00
  Final Equity                       :               106,420.50
  Total Return (%)                   :                    +6.42
  CAGR (%)                           :                   +18.25
  Sharpe Ratio                       :                     1.84
  Sharpe 95% CI                      :             [1.22, 2.45]
  Sortino Ratio                      :                     2.31
  Max Drawdown (%)                   :                     4.85
  Calmar Ratio                       :                     3.76
  Daily Win Rate (%)                 :                    58.20
-- TRADING ACTIVITY & CADENCE ------------------------------------
  Total Trades                       :                    1,248
  Buy Trades                         :                      625
  Sell Trades                        :                      623
  Trading Frequency (trades/day)     :                 1,248.00
  Average Trade Gap (steps)          :                    68.50
  Median Trade Gap (steps)           :                    45.00
  Min Trade Gap (steps)              :                        5
  Max Trade Gap (steps)              :                      420
  Market Exposure Time (%)           :                    82.15
-- EXPOSURE & RISK CONSTRAINTS -----------------------------------
  Mean Gross Leverage                :                     0.64
  Max Gross Leverage                 :                     0.98
  Gross Leverage Violations          :                        0
  Account Bankrupt (Capital <= 0)    :                        0
-- TURNOVER & TRANSACTION COSTS ----------------------------------
  Total Turnover (%)                 :                 1,840.25
  Total Turnover Value ($)           :             1,840,250.00
  Average Trade Value ($)            :                 1,474.56
  Total Commission Paid ($)          :                   184.03
  Total Slippage Paid ($)            :                   920.13
=================================================================
```

#### Key Diagnostic Insights:

- **`Sharpe 95% CI`:** Computed via a 1,000-iteration Stationary Block Bootstrap. If the lower bound is $\le 0.0$, the strategy does not demonstrate statistically significant alpha.
- **`Average & Median Trade Gap (steps)`:** Measures the frequency of trade adjustments. Extremely small gaps (e.g. 1 tick) indicate high churning that may incur excessive transaction costs.
- **`Market Exposure Time (%)`:** Percentage of the simulation duration where the portfolio held open positions.
- **`Total Commission & Slippage Paid`:** Cumulative friction paid. If friction exceeds your gross profit, your strategy needs filtering or wider thresholds.

---

### `results/summary.json`

Every metric displayed in the terminal is saved in machine-readable JSON format for automated grading and leaderboard aggregation.

---

### `results/trades.csv`

Every trade executed during the simulation is written to `results/trades.csv`:

```csv
step,ticker,side,shares,price,value,commission,slippage
5,ALPHA,BUY,1.4285,70035.00,100045.00,10.00,35.00
15,ALPHA,SELL,0.7142,70450.00,50315.39,5.03,17.61
```

Use this log to inspect execution prices, verify fill quantities, and debug slippage costs.

---

### 4-Panel Performance Tearsheet (`results/tearsheet.png`)

The backtester automatically renders a 4-panel diagnostic tearsheet:

1. **Panel 1: Equity Curve ($):** Tracks net portfolio liquidation value over time, noting final return and bankruptcy status.
2. **Panel 2: Underwater Drawdown (%):** Highlights drawdown depth and recovery periods.
3. **Panel 3: Gross Leverage Exposure:** Displays gross risk exposure against the strict $1.0\times$ boundary.
4. **Panel 4: Trade Execution Cadence:** Plots cumulative trade count over time with average and median step intervals.

---

## 10. Pre-Submission Sanity Checks & Best Practices

Before submitting your repository, execute:

```bash
python3 test_submission.py
```

The test runner performs 5 mandatory checks:

1. **File Structure Check:** Ensures all required files exist (`src/engine.py`, `strategy_base.py`, `backtester.py`, etc.).
2. **Class Definition & Inheritance:** Verifies that `ParticipantStrategy` implements `initialize()` and `on_bar()` and inherits from `BaseStrategy`.
3. **Track A Sanity Check:** Feeds synthetic multi-asset bars to test cross-sectional handling and leverage clamping.
4. **Track B Sanity Check:** Feeds high-frequency microstructure ticks to test OFI, VWAP, and single-asset handling.
5. **Non-Negative Capital & Leverage Rule Check:** Validates that the strategy respects non-negative cash limits and does not exceed gross leverage $\le 1.0$.
6. Include all research done, data analysis, model training, statistical analysis etc. code in src/research
7. The src/ directory must contain a pdf / markdown file explaining your rationale, analysis and strategy (max 10 pages)

### Common Pitfalls to Avoid:

- **Lookahead Bias:** Never store future rows or assume knowledge of future bars.
- **Overfitting & Excessive Churning:** High transaction costs can quickly destroy an otherwise promising signal. Check `Total Commission Paid` and `Total Slippage Paid`.
- **Exceeding Gross Leverage:** Make sure $\sum |w_i| \le 1.0$. The backtester clamps violations, but violations are logged in your final score.
- **Zero-Division Errors:** Protect against empty order books or zero volume bars when computing ratios (e.g. `taker_buy_volume / volume`).
- **Heavy Allocations in Loops:** Profile your code on Track B. Ensure your full backtest runs smoothly within competition timeout limits.

Good luck with your research! Focus on **statistical justification, risk management, and economic rationale**.
