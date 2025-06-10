# 📘 RCS-ML Forex/Gold Trading System

## Overview
A full machine learning system for predicting forex/gold movements using Relative Currency Strength (RCS) and other features. Built in Python, deployable via Streamlit (Hugging Face Spaces) or to MetaTrader 5 using ONNX.

## 📂 Folder Structure
```
project/
├── notebooks/
├── logs/
├── data/
├── models/
├── signal_logger.py
├── backtester.py
├── explainability.py
├── onnx_benchmark.py
├── rolling_labeling.py
├── feature_filtering.py
├── app.py
├── requirements.txt
└── .github/workflows/ci.yml
```

## ⚙️ Configuration
```yaml
symbols: ["XAUUSD", "EURUSD"]
lookback_window: 50
features:
  - rsi
  - macd
  - adx
  - roc
  - returns_lag
  - dxy
  - spx
use_feature_drift: true
use_correlation_filter: true
label_horizon: 10
label_method: binary
```

When using the Polygon provider, currency pairs may be written with or without
a `/` (e.g. `EURUSD` or `EUR/USD`). The data loaders automatically normalize
the symbol for Polygon's API. For the `yfinance` provider, append `=X` to
currency pairs (e.g. `EURUSD=X`).
When using `metatrader`, ensure the MetaTrader5 terminal is running.  Use
`--interval` or `--timeframe` to set the desired timeframe (e.g. `H1`, `M5`).
The loader selects the symbol in the terminal and downloads the requested
number of bars.


## 🔁 Feature Engineering
Supports:
- Technical indicators
- Lagged features
- Cross-asset (DXY, SPX)
- Macro (rates, inflation)
- Session-based (hour, weekday)

## 🔍 Feature Selection
- Manual
- Permutation Importance
- SHAP Ranking
- Greedy Forward
- Stability Filtering
- Correlation Filtering
- Drift Detection

## 🏷️ Rolling Labeling
- Binary, Ternary, Regression targets

## 🤖 Model Training
- CNN, LSTM, Trees
- ONNX + HDF5 output

## 🧪 Backtesting
- Metrics: Sharpe, Sortino, Max DD, Equity

## 📡 Signal Logging
JSON log to `logs/signals.log`

## 📈 Inference Benchmarking
Average ONNX latency (ms)

## 🔬 Explainability
SHAP summary plot to PNG

## 🧪 CI/CD Pipeline
Notebook execution + Docker build (GitHub Actions)

## 🌐 Streamlit App
- Signals table
- Metrics
- SHAP visualizations

## 🧠 Memory Storage (Future)
Redis/MCP optional integration
