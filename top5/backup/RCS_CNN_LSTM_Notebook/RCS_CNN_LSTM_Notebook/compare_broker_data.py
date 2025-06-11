import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from download_data import assess_quality, compare_dataframes

def align_dataframes(df1, df2):
    # Align on time, inner join
    df1 = df1.copy()
    df2 = df2.copy()
    df1["time"] = pd.to_datetime(df1["time"])
    df2["time"] = pd.to_datetime(df2["time"])
    merged = pd.merge(df1, df2, on="time", suffixes=("_1", "_2"), how="inner")
    return merged

def plot_overlay(merged, col, broker1, broker2, symbol, timeframe, outdir):
    plt.figure(figsize=(12, 5))
    plt.plot(merged["time"], merged[f"{col}_1"], label=broker1)
    plt.plot(merged["time"], merged[f"{col}_2"], label=broker2, alpha=0.7)
    plt.title(f"{symbol} {timeframe} {col} - Overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{symbol}_{timeframe}_{col}_overlay.png"))
    plt.close()

def plot_difference(merged, col, broker1, broker2, symbol, timeframe, outdir):
    diff = merged[f"{col}_1"] - merged[f"{col}_2"]
    plt.figure(figsize=(12, 5))
    plt.plot(merged["time"], diff)
    plt.title(f"{symbol} {timeframe} {col} Difference ({broker1} - {broker2})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{symbol}_{timeframe}_{col}_diff.png"))
    plt.close()

def plot_histogram(merged, col, broker1, broker2, symbol, timeframe, outdir):
    diff = merged[f"{col}_1"] - merged[f"{col}_2"]
    plt.figure(figsize=(8, 4))
    plt.hist(diff, bins=50, alpha=0.7)
    plt.title(f"{symbol} {timeframe} {col} Difference Histogram ({broker1} - {broker2})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{symbol}_{timeframe}_{col}_diff_hist.png"))
    plt.close()

def plot_scatter(merged, col, broker1, broker2, symbol, timeframe, outdir):
    plt.figure(figsize=(6, 6))
    plt.scatter(merged[f"{col}_1"], merged[f"{col}_2"], alpha=0.5, s=5)
    plt.xlabel(broker1)
    plt.ylabel(broker2)
    plt.title(f"{symbol} {timeframe} {col} Scatter")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{symbol}_{timeframe}_{col}_scatter.png"))
    plt.close()

def plot_rolling_stats(merged, col, broker1, broker2, symbol, timeframe, outdir, window=100):
    diff = merged[f"{col}_1"] - merged[f"{col}_2"]
    rolling_mean = diff.rolling(window).mean()
    rolling_std = diff.rolling(window).std()
    plt.figure(figsize=(12, 5))
    plt.plot(merged["time"], rolling_mean, label="Rolling Mean")
    plt.plot(merged["time"], rolling_std, label="Rolling Std")
    plt.title(f"{symbol} {timeframe} {col} Rolling Mean/Std of Difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{symbol}_{timeframe}_{col}_rolling.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare data quality/statistics between two brokers for a symbol/timeframe")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--broker1", required=True)
    parser.add_argument("--broker2", required=True)
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    base1 = f"{args.data_dir}/{args.symbol}_{args.timeframe}_2year_{args.broker1}.parquet"
    base2 = f"{args.data_dir}/{args.symbol}_{args.timeframe}_2year_{args.broker2}.parquet"
    outdir = "comparison_plots"
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading {base1}")
    df1 = pd.read_parquet(base1)
    print(f"Loading {base2}")
    df2 = pd.read_parquet(base2)

    print(f"\n--- Quality for {args.broker1} ---")
    assess_quality(df1)
    print(f"\n--- Quality for {args.broker2} ---")
    assess_quality(df2)
    print(f"\n--- Comparison ---")
    compare_dataframes(df1, df2, args.broker1, args.broker2)

    # Align and compare
    merged = align_dataframes(df1, df2)
    print(f"\nAligned rows: {len(merged)}")

    for col in ["open", "high", "low", "close"]:
        print(f"\n--- {col.upper()} Comparison ---")
        diff = merged[f"{col}_1"] - merged[f"{col}_2"]
        print(f"Mean difference: {diff.mean():.6f}")
        print(f"Std difference: {diff.std():.6f}")
        print(f"Min difference: {diff.min():.6f}")
        print(f"Max difference: {diff.max():.6f}")
        print(f"RMSE: {np.sqrt(np.mean(diff**2)):.6f}")
        print(f"MAE: {np.mean(np.abs(diff)):.6f}")
        corr = np.corrcoef(merged[f"{col}_1"], merged[f"{col}_2"])[0, 1]
        print(f"Pearson correlation: {corr:.6f}")

        # Plots
        plot_overlay(merged, col, args.broker1, args.broker2, args.symbol, args.timeframe, outdir)
        plot_difference(merged, col, args.broker1, args.broker2, args.symbol, args.timeframe, outdir)
        plot_histogram(merged, col, args.broker1, args.broker2, args.symbol, args.timeframe, outdir)
        plot_scatter(merged, col, args.broker1, args.broker2, args.symbol, args.timeframe, outdir)
        plot_rolling_stats(merged, col, args.broker1, args.broker2, args.symbol, args.timeframe, outdir)

    print(f"\nPlots saved as PNG files in the '{outdir}' directory.")

if __name__ == "__main__":
    main()
