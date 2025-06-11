from src.backtesting import run_backtest
import pandas as pd

results = run_backtest(
    model_path="exported_models/EURUSD_CNN_LSTM_20250610_211730.onnx",
    symbol="EURUSD",
    data_dir="data",
    lookback=50,
    initial_capital=100000,
    commission=0.0001,
    plot=True
)



from joblib import Parallel, delayed
import itertools

def run_backtest_with_params(lookback, commission):
    results = run_backtest(model_path="exported_models/EURUSD_CNN_LSTM_20250610_211730.onnx",
        symbol="EURUSD",
        data_dir="data",
        initial_capital=100000,
        lookback=lookback, 
        commission=0.0001,
        plot=False    
    )
    return {'lookback': lookback, 'commission': commission, 'sharpe': results['sharpe_ratio']}

lookbacks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
commissions = [0.0001]
param_combinations = list(itertools.product(lookbacks, commissions))

results = Parallel(n_jobs=-1)(delayed(run_backtest_with_params)(lb, comm) for lb, comm in param_combinations)
results_df = pd.DataFrame(results)
print(results_df)