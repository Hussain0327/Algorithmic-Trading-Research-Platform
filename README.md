# Trading Strategy Validation

A **research-grade** backtesting system for testing trading strategies against historical market data with rigorous statistical validation.

![Trading Platform Dashboard](screenshots/trade_log.png)

---

## Table of Contents

- [Quickstart](#quickstart)
- [Overview](#overview)
- [Features](#features)
- [Methodology Guardrails](#methodology-guardrails)
- [Installation](#installation)
- [Usage](#usage)
- [Strategies](#strategies)
- [Statistical Analysis](#statistical-analysis)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Quickstart

```bash
# Clone and setup (1 minute)
git clone https://github.com/Hussain0327/quant-backtesting-validation.git
cd quant-backtesting-validation
make setup

# Run demo backtest
make demo

# Expected output:
# ============================================================
# QUANTITATIVE TRADING RESEARCH FRAMEWORK
# ============================================================
# Running backtest with MA Crossover...
# [Training Period]
#   Return:       X.XX%
#   Sharpe:       X.XX
# [Test Period (Out-of-Sample)]
#   Return:       X.XX%
# ...
# Tests Passed: X/3
# Verdict: STRONG EVIDENCE OF EDGE (or similar)

# Generate HTML report
make report

# Launch interactive dashboard
make dashboard
```

---

## Overview

This project tests whether trading strategies actually outperform random chance. It combines backtesting with **rigorous statistical significance testing** to separate real edges from noise.

**Key insight:** A backtest showing 15% returns means nothing without knowing the probability that result occurred by chance.

**Core workflow:**

1. Fetch historical price data
2. Generate buy/sell signals using a strategy
3. Simulate trades with transaction costs
4. **Validate with statistical significance tests**
5. **Correct for multiple testing bias**

---

## Features

| Feature                     | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| Multiple Strategies         | MA Crossover, RSI, Momentum, Pairs Trading, Bollinger Bands  |
| **Walk-Forward Validation** | Tests across multiple time periods, not just one split       |
| **Deflated Sharpe Ratio**   | Corrects for multiple testing bias (Bailey & Lopez de Prado) |
| **Block Bootstrap**         | Preserves autocorrelation in resampling (Politis & Romano)   |
| Statistical Testing         | Bootstrap CI, permutation tests, Monte Carlo simulation      |
| **HTML Reports**            | Recruiter-friendly standalone reports with verdict           |
| Interactive Dashboard       | Professional Streamlit UI with dark theme                    |
| Cost Modeling               | Fixed, spread-based, and market impact models                |
| Assumptions Documentation   | Each test explicitly states what it assumes                  |

![Train vs Test Analysis](screenshots/analysis.png)

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/Hussain0327/quant-backtesting-validation.git
cd quant-backtesting-validation

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** If you get `ModuleNotFoundError`, make sure you've activated your virtual environment and run `pip install -r requirements.txt`.

---

## Usage

**Dashboard (Recommended):**

```bash
streamlit run app.py
```

![Charts View](screenshots/charts.png)

**Command Line:**

```bash
python main.py
```

---

## Strategies

Each strategy includes built-in documentation and recommended parameters:

| Strategy            | Type                  | Logic                                | Best For               |
| ------------------- | --------------------- | ------------------------------------ | ---------------------- |
| **MA Crossover**    | Trend-following       | Buy when short MA > long MA          | Trending markets       |
| **RSI**             | Mean reversion        | Buy oversold, sell overbought        | Range-bound markets    |
| **Momentum**        | Trend-following       | Trade in direction of recent returns | Strong trending stocks |
| **Pairs Trading**   | Statistical arbitrage | Mean reversion on z-score spread     | Correlated assets      |
| **Bollinger Bands** | Mean reversion        | Buy at lower band, sell at mean      | Volatility trading     |

---

## Statistical Analysis

Every backtest runs three significance tests:

| Test                 | Question                                               |
| -------------------- | ------------------------------------------------------ |
| **Sharpe CI**        | Is the Sharpe ratio significantly different from zero? |
| **Permutation Test** | Does it beat buy-and-hold?                             |
| **Monte Carlo**      | Does it beat random entry/exit?                        |

**Interpretation:**

- 3/3 pass → Strong evidence of edge
- 2/3 pass → Needs more investigation
- 0-1/3 pass → Likely noise

![Statistical Analysis](screenshots/statistics.png)

---

## Methodology Guardrails

This framework implements safeguards against common backtesting pitfalls:

### Look-Ahead Bias Prevention

| Protection         | Implementation                            |
| ------------------ | ----------------------------------------- |
| Signal generation  | Uses only data available at decision time |
| Rolling indicators | Computed with historical data only        |
| Train/test split   | Strict chronological separation           |

### Data Leakage Prevention

| Protection              | Implementation                           |
| ----------------------- | ---------------------------------------- |
| Chronological split     | Train period always precedes test period |
| Independent signals     | Signals regenerated for each period      |
| Walk-forward validation | Multiple folds to detect overfitting     |

### Multiple Testing Correction

When you test many strategies and pick the best, the reported Sharpe ratio is biased upward. We address this with:

| Method                      | Purpose                                                      |
| --------------------------- | ------------------------------------------------------------ |
| **Deflated Sharpe Ratio**   | Adjusts for number of trials (Bailey & Lopez de Prado 2014)  |
| **Walk-Forward Validation** | Tests across multiple time periods, not just one lucky split |
| **Explicit warnings**       | Dashboard shows "N trials tested" with adjusted p-values     |

### Statistical Test Assumptions

Each test documents what it assumes and what it doesn't account for:

```
┌─────────────────────────────────────────────────────────────┐
│ Block Bootstrap Sharpe CI                                    │
├─────────────────────────────────────────────────────────────┤
│ Assumes:   Stationary returns, finite variance              │
│ Preserves: Autocorrelation (block resampling)               │
│ Ignores:   Multiple testing, regime changes                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Known Limitations

### Data Quality

| Limitation            | Impact                                                    | Mitigation                                  |
| --------------------- | --------------------------------------------------------- | ------------------------------------------- |
| **Survivorship bias** | Yahoo Finance excludes delisted stocks, inflating returns | Documented; use premium data for production |
| **Adjusted prices**   | Backward-looking adjustments may change                   | Use unadjusted for intraday strategies      |
| **Data errors**       | Free data has known issues                                | Cross-validate with multiple sources        |

### Execution Assumptions

| Assumption               | Reality                        | Impact                |
| ------------------------ | ------------------------------ | --------------------- |
| Next-day close execution | Real fills vary by timing      | Understates slippage  |
| Unlimited liquidity      | Large orders move prices       | Ignores market impact |
| No partial fills         | Orders may not fill completely | Overstates capacity   |

### What This Framework Does NOT Solve

- **Live trading infrastructure** (broker APIs, latency, connectivity)
- **Real-time risk management** (position limits, stop-losses)
- **Regulatory compliance** (reporting, margin requirements)
- **Alternative data integration** (news, sentiment, fundamentals)

---

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI entry point
├── Makefile                # Setup and run commands
├── requirements.txt        # Pinned dependencies
│
├── strategies/
│   ├── base.py             # Abstract strategy class
│   ├── moving_average.py   # MA Crossover strategy
│   ├── rsi.py              # RSI strategy
│   ├── momentum.py         # Momentum strategy
│   └── pairs_trading.py    # Pairs Trading & Bollinger Bands
│
├── backtest/
│   ├── engine.py           # Core simulation logic
│   ├── costs.py            # Transaction cost models (fixed, spread, impact)
│   └── walk_forward.py     # Walk-forward validation
│
├── analytics/
│   ├── metrics.py          # Performance metrics (Sharpe, drawdown, etc.)
│   ├── significance.py     # Statistical tests with assumptions docs
│   ├── deflated_sharpe.py  # Multiple testing correction (DSR)
│   └── report.py           # HTML report generator
│
├── data/
│   ├── fetcher.py          # Yahoo Finance wrapper (with bias warnings)
│   └── database.py         # SQLite caching
│
├── reports/                # Generated HTML reports
│
├── tests/                  # Unit tests
│   ├── test_strategies.py
│   ├── test_engine.py
│   ├── test_metrics.py
│   └── test_significance.py
│
└── .streamlit/
    └── config.toml         # Dashboard theme configuration
```

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_strategies.py -v
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Tech Stack

| Component     | Technology    |
| ------------- | ------------- |
| Language      | Python 3.9+   |
| Data          | pandas, numpy |
| Statistics    | scipy         |
| Market Data   | yfinance      |
| Dashboard     | Streamlit     |
| Visualization | Plotly        |
| Testing       | pytest        |
