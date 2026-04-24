## Signal or noise? Intraday stock price prediction via Polymarket 

This repository implements a pipeline to study whether Polymarket prices contain predictive information about intraday stock returns. It is meant for the group paper written for the seminar in Financial Machine Learning at the ESE (MSc Financial Economics).

The project builds on the idea that market-implied probabilities (from betting markets, e.g. Polymarket) can be compared to theoretical fair values derived from financial models. We use the difference between the two as a signal to price stock returns.

---

### I. Data collection & orderbook construction
---

The core of the project is the Orderbook class, which retrieves and constructs a high-frequency dataset of trades from Polymarket.

1. Market discovery
   - Query Polymarket events using a date-based slug
   - Extract token IDs (UP / DOWN), condition IDs and metadata

2. Trade extraction
   - Pull order fill events via GraphQL in small time chunks
   - Identify trade direction (BUY / SELL), price and volume

3. Time alignment
   - Convert timestamps across timezones
   - Compute relative trading hours and time-to-expiry

---

### II. Collection of financial data
---

Polymarket data is merged with equity market data from Yahoo Finance:

- Daily data: open/close prices and stock direction
- Intraday data: prices and volatility
- Alignment ensures no look-ahead bias

---

### III. Feature engineering
---

1. Time window aggregation (`collapse_to_windows()`)
   - Aggregate trades into fixed intervals (5 minutes)
   - Compute prices, volume, trade count and order flow imbalance

2. $X^{poly}$ construction &rarr; because of mechanical price convergence at the end of the day

   $X^{poly} = \text{market-implied prob} − \text{Black–Scholes\ neutral prob}$

Interpretation:
- Positive → more bullish than theoretical value
- Negative → more bearish than theoretical value

---

### IV. Lead–lag analysis
---

- `check_lead_lag()`: correlation between sentiment and future returns
- `lead_lag_ccf()`: cross-correlation across multiple lags

Goal: test whether prediction markets lead stock markets (descriptive)

---

### V. Main model
---

1. OLS regression
   - Predict returns using $X^{poly}$ and trading features
   - Uses HAC standard errors

2. Logistic regression
   - Predict direction (up/down)
   - Metrics: accuracy, AUC, precision, recall

3. Elastic Net
   - Regularised regression (ridge + lasso)
   - Evaluated using out-of-sample $R^2$

---

### VI. Signal evaluation
---

- Analyse hit rates across sentiment strength
- Compare in-sample vs out-of-sample results
- Filter noisy/extreme observations

---

### VII. Backtesting
---

Function: `pro_backtest()`

1. Strategy logic
   - Trade when sentiment exceeds a threshold
   - Position size scales with confidence
   - Supports long-short, long-only, short-only

2. Market frictions
   - Transaction costs (bps)
   - Leverage
   - Capital constraints

3. Outputs
   - Total return
   - Sharpe ratio
   - Max drawdown
   - Win rate and hit rate

---

### VIII. Rolling & robustness analysis
---

- Rolling backtests over time windows
- Parameter sensitivity (confidence, costs)
- Strategy comparisons
- Intraday performance heatmaps

---

### IX. Setup
---

```python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
---

### X. Key assumptions
---

- Polymarket prices accurately reflect market beliefs
- Black-Scholes provides a neutral benchmark
- Deviations represent sentiment or inefficiency

---

### XI. Limitations
---

- Intraday data limited (~60 days)
- API rate limits slow data collection
- Simplified execution assumptions
- Results sensitive to parameters
