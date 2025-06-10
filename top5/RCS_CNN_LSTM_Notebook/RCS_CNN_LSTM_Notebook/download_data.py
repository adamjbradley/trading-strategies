import argparse
import asyncio

from async_data_loader import fetch_all_data, save_optimized


def assess_quality(df):
    """Basic data quality assessment for a DataFrame."""
    print("Data shape:", df.shape)
    print("Missing values per column:\n", df.isnull().sum())
    print("Number of duplicate rows:", df.duplicated().sum())
    if df.empty or df.shape[1] == 0:
        print("DataFrame is empty. No summary statistics available.")
    else:
        print("Summary statistics:\n", df.describe(include='all'))

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
