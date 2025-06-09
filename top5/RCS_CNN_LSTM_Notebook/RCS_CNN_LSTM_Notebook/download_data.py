import argparse
import asyncio

from async_data_loader import fetch_all_data
from data_loader import save_optimized, assess_quality, compare_dataframes


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

