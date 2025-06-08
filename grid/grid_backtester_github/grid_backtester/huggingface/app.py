
import gradio as gr
import pandas as pd
from strategy import run_grid_strategy
from metrics import compute_metrics
from utils import plot_equity
import matplotlib.pyplot as plt
import io

def backtest(file):
    df = pd.read_csv(file.name, parse_dates=['time'])
    df['ma'] = df['close'].rolling(20).mean()
    df['return'] = df['close'].pct_change()
    df['rolling_std'] = df['return'].rolling(10).std()
    df['vol_prediction'] = (df['high'] - df['low']) > df['rolling_std']
    df['vol_prediction'] = df['vol_prediction'].astype(int)
    df = df.dropna()

    final_balance, equity_df, trades_df = run_grid_strategy(df, 'EURUSD')
    metrics = compute_metrics(trades_df, equity_df)

    plt.figure()
    plt.plot(equity_df['time'], equity_df['equity'])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    summary = f"Final Balance: ${final_balance:,.2f}\n"
    for k, v in metrics.items():
        summary += f"{k}: {v:.4f}\n"

    return summary, buf

demo = gr.Interface(
    fn=backtest,
    inputs=gr.File(label="Upload EURUSD_H1.csv"),
    outputs=["text", "image"],
    title="📈 Grid Trading Backtester",
    description="Upload OHLC data to backtest a volatility-aware grid trading strategy."
)

if __name__ == "__main__":
    demo.launch()
