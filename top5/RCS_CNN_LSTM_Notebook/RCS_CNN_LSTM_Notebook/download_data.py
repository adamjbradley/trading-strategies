import argparse
import asyncio
import pandas as pd

from async_data_loader import fetch_all_data, save_optimized


def assess_quality(df):
    """Robust data quality assessment for a DataFrame, with a quality score."""
    print("Data shape:", df.shape)
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing)
    n_duplicates = df.duplicated().sum()
    print("Number of duplicate rows:", n_duplicates)
    if df.empty or df.shape[1] == 0:
        print("DataFrame is empty. No summary statistics available.")
        print("Quality score: 0/100 (empty data)")
        return 0

    print("Summary statistics:\n", df.describe(include='all'))

    # Initialize quality score
    score = 100
    reasons = []

    # Check for completeness and gaps
    n_gaps = 0
    max_gap = None
    if "time" in df.columns:
        times = pd.to_datetime(df["time"])
        times_sorted = times.sort_values()
        diffs = times_sorted.diff().dropna()
        most_common_diff = diffs.mode()[0] if not diffs.empty else None
        print(f"Most common time difference: {most_common_diff}")
        # Gaps: any diffs > 1.5x the mode
        if most_common_diff is not None:
            gap_threshold = most_common_diff * 1.5
            gaps = diffs[diffs > gap_threshold]
            n_gaps = len(gaps)
            max_gap = gaps.max() if n_gaps > 0 else None
            print(f"Number of gaps in time series: {n_gaps}")
            if n_gaps > 0:
                print("Gap details (first 5):")
                for idx in gaps.head().index:
                    prev_time = times_sorted.iloc[idx - 1]
                    next_time = times_sorted.iloc[idx]
                    gap_duration = gaps.loc[idx]
                    print(f"Gap from {prev_time} to {next_time}: {gap_duration}")
                score -= min(10, n_gaps)  # Deduct up to 10 points for gaps
                reasons.append(f"{n_gaps} time gaps")
                if max_gap is not None and max_gap > most_common_diff * 24:
                    score -= 5
                    reasons.append(f"large gap: {max_gap}")
        else:
            print("Could not determine time gaps (insufficient data).")

    # Check for price anomalies
    n_nonpos = 0
    n_missing = 0
    n_outliers = 0
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            n_nonpos_col = (df[col] <= 0).sum()
            n_nonpos += n_nonpos_col
            if n_nonpos_col > 0:
                print(f"Warning: Non-positive values found in {col} column.")
                score -= min(5, n_nonpos_col)
                reasons.append(f"{n_nonpos_col} non-positive {col}")
            n_missing_col = df[col].isnull().sum()
            n_missing += n_missing_col
            if n_missing_col > 0:
                print(f"Warning: Missing values found in {col} column.")
                score -= min(5, n_missing_col)
                reasons.append(f"{n_missing_col} missing {col}")
            # Outlier detection: z-score > 5
            z = (df[col] - df[col].mean()) / df[col].std()
            outliers = df[abs(z) > 5]
            n_outliers_col = len(outliers)
            n_outliers += n_outliers_col
            if n_outliers_col > 0:
                print(f"Warning: {n_outliers_col} outliers detected in {col} (z-score > 5).")
                print(outliers[[col, "time"]].head())
                score -= min(5, n_outliers_col)
                reasons.append(f"{n_outliers_col} outliers in {col}")

    # Check for splits/discontinuities (large jumps)
    n_jumps = 0
    if "close" in df.columns:
        close = df["close"].astype(float)
        jumps = close.pct_change().abs()
        big_jumps = jumps[jumps > 0.1]  # >10% jump
        n_jumps = len(big_jumps)
        print(f"Number of >10% jumps in close price: {n_jumps}")
        if n_jumps > 0:
            print("Jump details (first 5):")
            print(df.loc[big_jumps.index, ["time", "close"]].head())
            score -= min(5, n_jumps)
            reasons.append(f"{n_jumps} >10% jumps")

    # Duplicates
    if n_duplicates > 0:
        score -= min(5, n_duplicates)
        reasons.append(f"{n_duplicates} duplicate rows")

    # Clamp score to [0, 100]
    score = max(0, min(100, score))
    print(f"Quality score: {score}/100")
    if reasons:
        print("Quality deductions:", "; ".join(reasons))
    else:
        print("No quality issues detected.")
    return score

def compare_dataframes(df1, df2, provider1, provider2):
    """Basic comparison between two DataFrames."""
    print(f"Comparing {provider1} vs {provider2}:")
    print(f"Shape: {provider1}: {df1.shape}, {provider2}: {df2.shape}")
    if df1.shape != df2.shape:
        print("DataFrames have different shapes.")
    else:
        diff = (df1 != df2).sum().sum()
        print(f"Number of differing values: {diff}")
    print("Missing values in first DataFrame:\n", df1.isnull().sum())
    print("Missing values in second DataFrame:\n", df2.isnull().sum())

def main():
    parser = argparse.ArgumentParser(description="Manual data downloader")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--interval", default="1min")
    parser.add_argument("--outputsize", type=int, default=500)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--broker", default="default")
    parser.add_argument("--compare_provider")
    parser.add_argument("--compare_key", default="")
    args = parser.parse_args()

    df_dict = asyncio.run(
        fetch_all_data(
            [args.symbol],
            args.provider,
            args.api_key,
            interval=args.interval,
            outputsize=args.outputsize,
            start=args.start,
            end=args.end,
            broker=args.broker,
        )
    )
    df = df_dict[args.symbol]
    save_optimized(df, args.symbol, args.provider)
    assess_quality(df)

    if args.compare_provider:
        df_dict2 = asyncio.run(
            fetch_all_data(
                [args.symbol],
                args.compare_provider,
                args.compare_key,
                interval=args.interval,
                outputsize=args.outputsize,
                start=args.start,
                end=args.end,
            )
        )
        df2 = df_dict2[args.symbol]
        save_optimized(df2, args.symbol, args.compare_provider)
        assess_quality(df2)
        compare_dataframes(df, df2, args.provider, args.compare_provider)


if __name__ == "__main__":
    main()
