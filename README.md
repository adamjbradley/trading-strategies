# Trading Strategies
Trading Strategies

Supported provider strings for `fetch_all_data` are `twelvedata`, `polygon`,
`yfinance` and `metatrader`.

This repository aggregates two independent algorithmic trading projects:

- **Grid Trading Strategy Framework** – a forex grid approach with ONNX-based volatility prediction located under [`grid/grid_backtester_github/grid_backtester`](grid/grid_backtester_github/grid_backtester).
- **RCS-ML Forex/Gold Trading System** – a CNN/LSTM machine learning pipeline for currency strength analysis under [`top5/RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook`](top5/RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook).

## Quick Setup
1. Clone this repository and install Python 3.8+.
2. Install dependencies for each project using the respective `requirements.txt` files.
   ```bash
   pip install -r grid/grid_backtester_github/grid_backtester/requirements.txt
   pip install -r top5/RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook/requirements.txt
   ```
3. Explore the Jupyter notebooks in each project for detailed examples:
   - `GridBacktestNotebook.ipynb` for the grid strategy
   - `RCS_CNN_LSTM.ipynb` for the RCS system
4. Run the provided backtesting scripts to evaluate strategies:
   ```bash
   python grid/grid_backtester_github/grid_backtester/run_backtest.py
   python top5/RCS_CNN_LSTM_Notebook/RCS_CNN_LSTM_Notebook/backtest.py
   ```
For more details, refer to the READMEs inside each project directory.