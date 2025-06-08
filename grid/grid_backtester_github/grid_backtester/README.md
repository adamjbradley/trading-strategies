
# Grid Trading Strategy Framework

This project provides a complete forex grid trading framework with:
- Python backtesting engine
- ONNX-based ML volatility prediction
- MetaTrader 5 Expert Advisor
- Full automation pipeline

## 📦 Contents

- `strategy.py` — Grid strategy logic
- `metrics.py` — Sharpe ratio, drawdown, win rate
- `utils.py` — Data loading and equity curve plotting
- `run_backtest.py` — Multi-symbol backtesting script
- `download_data.py` — Pull OHLC data from MetaTrader 5
- `train_model.py` — Train ML model and export to ONNX
- `AdaptiveGridEA.mq5` — Real-time EA in MetaTrader 5
- `GridBacktestNotebook.ipynb` — Interactive notebook

## ✅ Setup Instructions

1. Install dependencies:
```bash
pip install MetaTrader5 pandas numpy matplotlib scikit-learn skl2onnx
```

2. Run `download_data.py` to get historical OHLC data.

3. Run `train_model.py` to build and export `volatility_predictor.onnx`.

4. Run backtest:
```bash
python run_backtest.py
```

5. Use `AdaptiveGridEA.mq5` in MetaTrader 5. Make sure to output model predictions as:
```
vol_prediction.csv: 1,<grid_pips>,<lot_multiplier>
```

6. For visual exploration, use `GridBacktestNotebook.ipynb`.

Enjoy smarter grid trading!


7. Use conda to manage environments! grid_trading env
