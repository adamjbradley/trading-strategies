import numpy as np
import pandas as pd

def backtest_signals(df, signal_col="signal", price_col="close", initial_cash=10000, position_size=1.0):
    df = df.copy()
    df["position"] = df[signal_col].shift(1).fillna(0)
    df["returns"] = df[price_col].pct_change().fillna(0)
    df["strategy_returns"] = df["position"] * df["returns"]

    df["equity"] = (1 + df["strategy_returns"]).cumprod() * initial_cash

    return df

def compute_risk_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret != 0 else 0

    downside_ret = returns[returns < 0]
    sortino = mean_ret / downside_ret.std() * np.sqrt(252) if downside_ret.std() != 0 else 0

    max_dd = (equity_curve / equity_curve.cummax() - 1).min()

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "final_equity": equity_curve.iloc[-1]
    }
